"""Cabinet pair-data viewer: ntrl-demo training pairs (x0, x1) as grasp-point pairs. Data sets: the in-cabinet pairs of
joint-range Test 3, Test 3's mixtures, the in-area pairs and the old recipe.

Data sets (release coordinates x = q / q_range in all):
  cabinet  Test 3's in-cabinet pairs (results/registered_20261031T210851Z-ac3c7c7c/data/cabinet): in-area pairs whose
           x0 or x1 grasp point is also inside the cupboard's outer box
  inarea   the joint-range test's pairs (results/registered_20261031T192723Z-772f8677/data/full): both grasp points in the
           scene's sampling box (the test area)
  old      the registered ntrl-demo cabinet baseline's pairs (results/registered_20261019T034237Z-30d4d872/data, trained
           by registered_20261025T065945Z-3c53b360): x0's grasp point in the box, x1 a random step anywhere
For each, a fixed random subset of --count pairs (seed --seed; the page shows the first N): grasp-frame positions and
arm joint origins of x0 and x1 (forward kinematics of the release URDF), their speed labels and the joint-space step
|x1 - x0|. Over ALL pairs: histograms of the step, test-area, cupboard and contact shares. The same over the exact
training data of each Test 3 mixture run (the first round(f * 200k) in-cabinet pairs and the first 200k - that in-area
pairs, as mix.py), with the run's planning results: totals and success split by the number of query endpoints in the
cupboard (as the README's Test 3 table). Also the start-goal distances of the 600 frozen test queries. Read-only;
writes pairs_viewer.html.

  ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/build_pairs_viewer.py
"""
import argparse, base64, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT)]
from experiments.face_speed import robot as R

SETS = {'cabinet': ROOT/'results/registered_20261031T210851Z-ac3c7c7c/data/cabinet', 'inarea': ROOT/'results/registered_20261031T192723Z-772f8677/data/full',
        'old': ROOT/'results/registered_20261019T034237Z-30d4d872/data'}
MIX_RUNS = {0.0: '211045Z-3afa500d', 0.1: '211100Z-892183cc', 0.2: '212120Z-7fb82c6c', 0.3: '212046Z-cd780569', 0.4: '213209Z-878950bf', 0.5: '213030Z-ce160baa',
            0.6: '214255Z-8382a03f', 0.7: '214028Z-00ac905b', 0.8: '215340Z-43d58912', 0.9: '215012Z-41858b8d', 1.0: '220420Z-9105ec65'}
MIX_PAIRS = 200000
METHODS = ('ntrl_mppi', 'matched_mppi', 'native_rk4')
SCENE = ROOT/'experiments/cabinet/scenes/cabinet.yml'
TESTS = ROOT/'experiments/cabinet/tests'
ARM = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', R.EE_LINK]
BINS = np.linspace(0, .6, 61)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--count', type=int, default=4000); p.add_argument('--seed', type=int, default=20261031)
    p.add_argument('--output', type=Path, default=HERE/'pairs_viewer.html'); a = p.parse_args()
    norm = np.load(R.NORMALIZATION); qr = norm['q_range'].astype(np.float64); kin = R.Kinematics()
    scene = yaml.safe_load(SCENE.read_text()); box = np.array([scene['sampling'][k] for k in 'xyz'])
    panels = [c for c in scene['cuboids'] if c['name'].startswith('shelf_')]
    cup = np.stack([np.min([np.subtract(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0),
                    np.max([np.add(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0)], 1)
    inside = lambda pos, b: ((pos >= b[:, 0]) & (pos <= b[:, 1])).all(1)

    def frames(x, links):                    # release coordinates [n, 6] -> {link: [n, 3]}
        out = {l: [] for l in links}
        with torch.no_grad():
            for s in range(0, len(x), 50000):
                t = kin.transforms(torch.as_tensor(np.asarray(x[s:s+50000], np.float64)*qr), links)
                for l in links: out[l].append(t[l][:, :3, 3].numpy())
        return {l: np.concatenate(v) for l, v in out.items()}

    tm = json.loads((TESTS/'suite_manifest.json').read_text()); suites = [np.load(TESTS/'suites'/s['file'], allow_pickle=True) for s in tm['suites']]
    tasks = [s['task'] for s in tm['suites']]
    # endpoints of each query in the cupboard (0, 1 or 2), from the suites' scorer positions, as the README's Test 3 table
    qk = {t: sum(inside(np.stack([z[f'ee_{c}_{w}'] for c in 'xyz'], 1), cup).astype(int) for w in ('start', 'goal')) for t, z in zip(tasks, suites)}
    k_all = np.concatenate([qk[t] for t in tasks])
    qs = np.concatenate([s['starts_rad'] for s in suites])/qr; qg = np.concatenate([s['goals_rad'] for s in suites])/qr
    test_gap = np.linalg.norm(qg-qs, axis=1); test_ends = frames(np.concatenate([qs, qg]), [R.EE_LINK])[R.EE_LINK]
    data = dict(links=ARM, count=a.count, bins=BINS.tolist(), test=dict(hist=(np.histogram(test_gap, BINS)[0]/len(test_gap)).tolist(), median=float(np.median(test_gap)),
                n=len(test_gap), ends=b64(test_ends), k_counts=[int((k_all == j).sum()) for j in range(3)]),
                boxes=[dict(name=c['name'], center=c['pose'][:3], dims=c['dims']) for c in scene['cuboids']],
                sampling=scene['sampling'], cupboard=cup.tolist(), sets={}, mix=[])

    def stats(step, e0, e1, k0, k1):         # k0 / k1: x0 / x1 at the floor speed (contact)
        n = len(step); c0, c1 = inside(e0, cup), inside(e1, cup); cp = int(c0.sum()+c1.sum())
        return dict(pairs=n, x1_in_box=float(inside(e1, box).mean()), x0_in_box=float(inside(e0, box).mean()), x0_cupboard=float(c0.mean()),
                    x1_cupboard=float(c1.mean()), pair_cupboard=float((c0 | c1).mean()), points_cupboard=cp/(2*n),
                    cupboard_contact=float(((k0 & c0).sum()+(k1 & c1).sum())/max(cp, 1)), step_median=float(np.median(step)), step_p90=float(np.quantile(step, .9)),
                    longer_than_test_median=float((step > np.median(test_gap)).mean()), x1_floor_speed=float(k1.mean()),
                    hist=(np.histogram(step, BINS)[0]/n).tolist())

    keep = {}
    for name, folder in SETS.items():
        meta = json.loads((folder/'complete.json').read_text())
        X = np.load(folder/'sampled_points.npy', mmap_mode='r'); S = np.asarray(np.load(folder/'speed.npy', mmap_mode='r')); n = len(X)
        step = np.linalg.norm(np.asarray(X[:, 6:])-np.asarray(X[:, :6]), axis=1)
        e0 = frames(X[:, :6], [R.EE_LINK])[R.EE_LINK]; e1 = frames(X[:, 6:], [R.EE_LINK])[R.EE_LINK]
        floor = float(S.min()); assert abs(floor-.1) < 1e-4, (name, floor)
        k0, k1 = S[:, 0] <= floor+1e-6, S[:, 1] <= floor+1e-6
        full = stats(step, e0, e1, k0, k1); keep[name] = (step, e0, e1, k0, k1)
        pick = np.random.default_rng(a.seed).choice(n, a.count, replace=False)
        x0, x1 = np.asarray(X[pick, :6]), np.asarray(X[pick, 6:]); f0 = frames(x0, ARM); f1 = frames(x1, ARM)
        arm0 = np.stack([f0[l] for l in ARM], 1); arm1 = np.stack([f1[l] for l in ARM], 1)
        data['sets'][name] = dict(meta_pairs=meta.get('pairs'), full=full, index=pick.tolist(), ee0=b64(arm0[:, -1]), ee1=b64(arm1[:, -1]), arm0=b64(arm0), arm1=b64(arm1),
                                  q0=b64(x0*qr), q1=b64(x1*qr), s0=b64(S[pick, 0]), s1=b64(S[pick, 1]), step=b64(step[pick]))
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in full.items() if k != 'hist'}, flush=True)

    A, B = keep['cabinet'], keep['inarea']
    for f, rid in MIX_RUNS.items():
        run = ROOT/'results'/f'registered_20261031T{rid}'; st = json.loads((run/'status.json').read_text()); cfg = st['settings']
        assert st['complete'] and st['mode'] == 'mix' and not cfg['smoke'] and cfg['fraction'] == f and cfg['pairs'] == MIX_PAIRS, run
        assert Path(cfg['cabinet']).parents[1].name == SETS['cabinet'].parents[1].name and Path(cfg['data']).parents[1].name == SETS['inarea'].parents[1].name, run
        n_cab = int(round(f*MIX_PAIRS)); part = [np.concatenate([u[:n_cab], v[:MIX_PAIRS-n_cab]]) for u, v in zip(A, B)]
        ev = json.loads((run/'evaluation/evaluation.json').read_text()); by_k = {}
        for meth in METHODS:
            sc = [x for x in ev['scores'] if x['method'] == meth]; seeds = len({x['planning_seed'] for x in sc}); succ = {t: np.zeros(len(qk[t])) for t in tasks}
            for x in sc: succ[x['task']][np.array(x['success_ids'], int)] += 1/seeds
            s_all = np.concatenate([succ[t] for t in tasks]); by_k[meth] = [float(s_all[k_all == j].mean()) for j in range(3)]
        data['mix'].append(dict(fraction=f, run=run.name, stats=stats(*part), totals={m: st['totals'][m]['success'] for m in METHODS}, by_k=by_k))
        print('mix', f, data['mix'][-1]['totals'], [round(100*v, 1) for v in by_k['ntrl_mppi']], round(data['mix'][-1]['stats']['pair_cupboard'], 3), flush=True)
    html = (HERE/'pairs_template.html').read_text().replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
    a.output.write_text(html); print(a.output, len(html)//1024, 'KB', 'test gap median', round(data['test']['median'], 3), 'queries per k', data['test']['k_counts'])


if __name__ == '__main__': main()
