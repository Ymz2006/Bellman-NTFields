"""Build the 3-D path viewer of the LBS arm study: report/viewer/arm_paths.html + one data file per scene.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs_arm/viewer.py

For every scene in viewer_scenes.json ({key: {name, prepared, runs: [[key, name, field run], ...]}}) and every frozen query: the suite's
certified path and each run's plan of planning seed 0, resampled to FRAMES configurations by joint-space arc length and stored as int16
(normalized joints x 60000). A plan that did not reach its goal is shown up to its last planned configuration (the planner's appended
jump to the goal is dropped). Outcomes are this study's copy of the scorer's rule (core.score), as in the report's failure table.
The page does the forward kinematics itself (the chain and the collision spheres of arm.py); three test configurations are shipped so
the page can check its kinematics against arm.py's.
"""
import base64
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import arm as A  # noqa: E402
import arm12 as A12  # noqa: E402
import core as C  # noqa: E402

FRAMES = 64
SCALE = 60000.0
OUT = HERE / 'report/viewer'
TASK = dict(free_to_free='free → free', free_to_compartment='free → compartment', compartment_to_free='compartment → free',
            compartment_to_compartment='compartment → another compartment', same_compartment='inside one compartment', anywhere='anywhere → anywhere')


def resample(path, n=FRAMES):
    """(n, 6) points evenly spaced by arc length along a polyline (m, 6)."""
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    path = path[keep]
    if len(path) == 1:
        return np.repeat(path, n, 0)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, path[:, j]) for j in range(path.shape[1])], 1)


def scene_file(name):
    if name.startswith('real_world'):
        return HERE / 'scenes' / (name + '.yml')
    return ROOT / 'experiments/cabinet/scenes' / (name + '.yml')


def build(key, cfg, dev):
    prepared = ROOT / cfg['prepared']
    multi = json.loads((prepared / 'scene.json').read_text())['describe'].get('dim', 6) > 6            # several arms (arm12.ArmSet)
    scene, info = (A12 if multi else A).load_scene(prepared, dev)
    ND = len(scene.lo)
    step_rad = info.get('score_step_rad', 0.002)                                                       # the resolution of the run's own count
    Q = np.load(prepared / 'queries.npz', allow_pickle=True)
    S, G = Q['start'], Q['goal']
    qr = scene.q_range.cpu().numpy()
    tests = Path(info['tests'])
    man = json.loads((tests / 'suite_manifest.json').read_text())
    wit = []
    for item in man['suites']:
        z = np.load(tests / 'suites' / item['file'], allow_pickle=True)
        wit += [np.asarray(w, np.float64) / qr for w in z['witness_trajectories']]
    assert len(wit) == len(S) and all(np.abs(w[0] - S[i]).max() < 1e-4 and np.abs(w[-1] - G[i]).max() < 1e-4 for i, w in enumerate(wit))
    n = len(S)
    names = [('cert', 'certified path')] + [(r[0], r[1]) for r in cfg['runs']]
    data = np.zeros((n, len(names), FRAMES, ND), np.int16)
    status = np.zeros((n, len(names)), np.int8)                      # 0 solved, 1 arrived on a path in contact, 2 did not arrive
    length = np.zeros((n, len(names)), np.float32)
    for i in range(n):
        data[i, 0] = np.round(resample(wit[i]) * SCALE)
        length[i, 0] = np.linalg.norm(np.diff(wit[i], axis=0), axis=1).sum()
    runs = []
    for k, (rk, rname, rdir) in enumerate(cfg['runs'], start=1):
        rd = ROOT / rdir
        fr = json.loads((rd / 'results.json').read_text())
        raw = np.load(rd / 'paths_seed0.npy', allow_pickle=True)
        paths = [np.asarray(p, np.float64) / qr for p in raw]
        near = 2 * fr['planner']['tol'] + fr['planner']['step']
        arrived = np.array([len(p) >= 2 and np.linalg.norm(p[-2] - G[i]) <= near for i, p in enumerate(paths)])
        rows = C.score(scene, [torch.tensor(p, dtype=torch.float32, device=dev) for p in paths], arrived, step_rad=step_rad)
        ok = np.array([r['success'] for r in rows])
        status[:, k] = np.where(ok, 0, np.where(arrived, 1, 2))
        for i, p in enumerate(paths):
            shown = p if arrived[i] or ok[i] else p[:-1]
            data[i, k] = np.round(resample(shown) * SCALE)
            length[i, k] = np.linalg.norm(np.diff(shown, axis=0), axis=1).sum()
        runs.append(dict(key=rk, name=rname, id=rd.name[len('registered_'):], solved=int(ok.sum()), contact=int((~ok & arrived).sum()), stopped=int((~ok & ~arrived).sum()),
                         own_seed0=fr['plan']['0']['success']))
        print(key, rk, 'solved', int(ok.sum()), 'in contact', int((~ok & arrived).sum()), 'did not arrive', int((~ok & ~arrived).sum()), '; pipeline counted', fr['plan']['0']['success'])
    doc = yaml.safe_load(scene_file(info['tests_scene']).read_text())
    cub = [dict(name=c['name'], c=[float(v) for v in c['pose'][:3]], d=[float(v) for v in c['dims']]) for c in doc['cuboids']]
    assert np.allclose(np.array([c['c'] for c in cub]), scene.cc.cpu().numpy(), atol=1e-6)
    m = np.load(prepared / 'robot_model.npz', allow_pickle=False)
    g = torch.Generator('cpu').manual_seed(3)
    xt = ((torch.rand(3, ND, generator=g) - 0.5) * 0.6).to(dev)
    pt, et = scene.spheres(xt)
    pt, et = pt.reshape(3, -1, 3), et.reshape(3, -1, 3)                                              # (3, arms x spheres, 3), (3, arms, 3)
    bases = None
    if multi:
        bases = np.tile(np.eye(4), (scene.A, 1, 1))
        bases[:, :3, :3], bases[:, :3, 3] = scene.base_R.cpu().numpy(), scene.base_t.cpu().numpy()
    ia, ib = scene.pair_a.cpu().numpy(), scene.pair_b.cpu().numpy()
    meta = dict(key=key, name=cfg['name'], scene=info['tests_scene'], frames=FRAMES, scale=SCALE, paths=[dict(key=a, name=b) for a, b in names], runs=runs,
                tasks=[TASK[str(t)] for t in Q['task_names']], task=Q['task'].astype(int).tolist(), line=Q['line'].astype(int).tolist(),
                status=status.tolist(), length=np.round(length, 4).tolist(), cuboids=cub,
                box=scene.box.cpu().numpy().round(5).tolist(), cupboard=scene.cupboard.cpu().numpy().round(5).tolist(),
                compartments=[dict(name=str(nm), box=np.round(b, 5).tolist()) for nm, b in zip(m['compartment_names'], m['compartments'])],
                arm=dict(T0=scene.T0.cpu().numpy().astype(float).round(9).tolist(), O=scene.O.cpu().numpy().astype(float).round(9).tolist(), frame=scene.frame.cpu().numpy().astype(int).tolist(),
                         local=scene.local.cpu().numpy().astype(float).round(6).tolist(), radius=scene.radius.cpu().numpy().astype(float).round(6).tolist(),
                         ee_local=scene.ee_local.cpu().numpy().astype(float).round(6).tolist(), q_range=qr.astype(float).tolist(), ignore=m['ignore'].astype(int).tolist(),
                         **({'bases': bases.round(9).tolist()} if multi else {}),
                         pairs=np.stack([ia, ib], 1).astype(int).tolist()),
                check=dict(x=xt.cpu().numpy().astype(float).tolist(), spheres=pt.cpu().numpy().astype(float).round(6).tolist(), ee=et.cpu().numpy().astype(float).round(6).tolist()),
                data=base64.b64encode(data.astype('<i2').tobytes()).decode())
    f = OUT / f'scene_{key}.json'
    f.write_text(json.dumps(meta, separators=(',', ':')))
    print(f, f.stat().st_size // 1024, 'KB')
    return dict(key=key, name=cfg['name'], file=f.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / 'viewer_scenes.json').read_text())
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    index = [build(k, v, dev) for k, v in cfg.items()]
    html = (HERE / 'viewer_template.html').read_text()
    assert html.count('/*SCENES*/[]') == 1
    (OUT / 'arm_paths.html').write_text(html.replace('/*SCENES*/[]', json.dumps(index)))
    print(OUT / 'arm_paths.html')


if __name__ == '__main__':
    main()
