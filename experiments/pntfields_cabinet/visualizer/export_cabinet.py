"""Export Task 2 (P-NTFields vs ntrl-demo on the user's cabinets) for the path viewer.

For each scene, a few frozen queries per suite with the certified witness and the saved paths of the registered
P-NTFields run (experiments/pntfields_cabinet) and the ntrl-demo run it is compared with (experiments/ntrl_cabinet):
both fields under ntrl-demo's MPPI and the release MPPI (planning seed 0) and the native RK4, plus P-NTFields' own
planner. Verdicts are the release scorer's (the saved .score.json). For colouring, every path is resampled at
0.002 rad and each frame gets the release sphere model's environment / self clearance (experiments/face_speed/
clearance.py, cover 'split'); a frame is flagged with the scorer's thresholds (cabinet < 5 mm, self < 2 mm).
Queries are picked so both fields under the same planner (ntrl-demo MPPI, seed 0) disagree in both directions,
agree on success and agree on failure. Read-only; writes one JSON file.

  ~/.venvs/bellman-ntfield/bin/python experiments/pntfields_cabinet/visualizer/export_cabinet.py
"""
import argparse, base64, json, math, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import trimesh
import yaml
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.face_speed import robot as R
from experiments.face_speed.clearance import PieceClearance, SELF_SCALE

# scene -> (tests dir, P-NTFields run, ntrl-demo run)
SCENES = {'cabinet': ('tests', '20261026T080627Z-53c1dd59', '20261026T041234Z-664b5d50'),
          'cabinet_pi': ('tests_pi', '20261026T110153Z-44e64765', '20261026T041504Z-379bcde5'),
          'cabinet_w25': ('tests_w25', '20261026T140052Z-b8746f48', '20261026T022529Z-b25f368e'),
          'cabinet_w20': ('tests_w20', '20261026T165453Z-9a9b9b4d', '20261026T022740Z-c32709b5'),
          'cabinet_w15': ('tests_w15', '20261026T194913Z-656fe377', '20261026T023140Z-b00193b9')}
SOURCES = {'witness': 'Certified witness path',
           'pnt_plan': 'P-NTFields field, P-NTFields planner',
           'pnt_ntrl': 'P-NTFields field, ntrl-demo MPPI',
           'ntrl_ntrl': 'ntrl-demo field, ntrl-demo MPPI',
           'pnt_mppi': 'P-NTFields field, release MPPI',
           'ntrl_mppi': 'ntrl-demo field, release MPPI',
           'pnt_rk4': 'P-NTFields field, native RK4',
           'ntrl_rk4': 'ntrl-demo field, native RK4'}
LABEL = {'pnt_plan': ('pnt', 'pnt_plan_{t}'), 'pnt_ntrl': ('pnt', 'pnt_ntrl_mppi_{t}_seed0'), 'ntrl_ntrl': ('ntrl', 'metric_ntrl_mppi_{t}_seed0'),
         'pnt_mppi': ('pnt', 'pnt_mppi_{t}_seed0'), 'ntrl_mppi': ('ntrl', 'metric_mppi_{t}_seed0'), 'pnt_rk4': ('pnt', 'pnt_rk4_{t}'), 'ntrl_rk4': ('ntrl', 'metric_rk4_{t}')}
TOTALS = {'pnt_plan': ('pnt', 'pnt_plan'), 'pnt_ntrl': ('pnt', 'ntrl_mppi'), 'ntrl_ntrl': ('ntrl', 'ntrl_mppi'), 'pnt_mppi': ('pnt', 'matched_mppi'),
          'ntrl_mppi': ('ntrl', 'matched_mppi'), 'pnt_rk4': ('pnt', 'native_rk4'), 'ntrl_rk4': ('ntrl', 'native_rk4')}
PER_SUITE = 4
OBS_MARGIN, SELF_MARGIN, STEP = .005, .002, .002   # score_cross_suite.py defaults (its step is 0.001 rad)
MAX_FRAMES = 160


def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def urdf_joints(path, moving):
    out = []
    for j in ET.parse(path).getroot().findall('joint'):
        o = j.find('origin'); ax = j.find('axis'); t = np.eye(4)
        if o is not None:
            t[:3, :3] = R.rpy_matrix(*[float(x) for x in o.get('rpy', '0 0 0').split()]).numpy(); t[:3, 3] = [float(x) for x in o.get('xyz', '0 0 0').split()]
        out.append(dict(name=j.get('name'), parent=j.find('parent').get('link'), child=j.find('child').get('link'), origin=t.reshape(-1).round(9).tolist(),
                        axis=[float(x) for x in (ax.get('xyz') if ax is not None else '0 0 1').split()], index=moving.index(j.get('name')) if j.get('name') in moving else -1))
    return out


def robot_meshes():
    meshes = []
    for link in ET.parse(R.URDF).getroot().findall('link'):
        col = link.find('collision'); mesh = None if col is None else col.find('geometry/mesh')
        if mesh is None: continue
        m = trimesh.load(R.URDF.parent/mesh.get('filename'), force='mesh'); m.apply_scale([float(x) for x in mesh.get('scale', '1 1 1').split()]); m.merge_vertices()
        o = col.find('origin'); t = np.eye(4)
        if o is not None:
            t[:3, :3] = R.rpy_matrix(*[float(x) for x in o.get('rpy', '0 0 0').split()]).numpy(); t[:3, 3] = [float(x) for x in o.get('xyz', '0 0 0').split()]
        meshes.append(dict(link=link.get('name'), origin=t.reshape(-1).round(9).tolist(), positions=b64(m.vertices.astype(np.float32)),
                           indices=b64(m.faces.astype(np.uint32)), triangles=int(len(m.faces))))
    return meshes


def dense(path, res=.01):
    """Resampled path and, per waypoint, the index range of dense configurations on the segment into it."""
    out, ends = [path[:1]], [1]
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, int(math.ceil(np.abs(b-a).max()/res))); out.append(a+(b-a)*(np.arange(1, k+1)[:, None]/k)); ends.append(ends[-1]+k)
    return np.concatenate(out), np.array(ends)


def judge(pc, path):
    d, ends = dense(np.asarray(path, np.float64), STEP)
    with torch.no_grad():
        q = (torch.tensor(d, dtype=torch.float32)-pc.robot.q_center)/pc.robot.q_range
        pieces = torch.cat([pc.pieces(q[i:i+4096]) for i in range(0, len(q), 4096)]).numpy()
    env, slf = pieces[:, 0], pieces[:, 1]/SELF_SCALE
    starts = np.r_[0, ends[:-1]]
    seg_env = np.array([env[s:e].min() for s, e in zip(starts, ends)]); seg_self = np.array([slf[s:e].min() for s, e in zip(starts, ends)])
    return seg_env, seg_self


def thin(n, keep=()):
    if n <= MAX_FRAMES: return np.arange(n)
    return np.unique(np.r_[np.linspace(0, n-1, MAX_FRAMES).round().astype(int), list(keep)])


def segment_min(values, keep):
    return np.array([values[(keep[n-1]+1 if n else 0):keep[n]+1].min() for n in range(len(keep))])


def export_scene(scene, tests, runs, pc):
    paths = {k: ROOT/'results'/f'registered_{v}'/'evaluation' for k, v in zip(['pnt', 'ntrl'], runs)}
    evaluation = {k: json.loads((p/'evaluation.json').read_text()) for k, p in paths.items()}
    manifest = json.loads((ROOT/'experiments/cabinet'/tests/'suite_manifest.json').read_text())
    for e in evaluation.values(): assert e['tests_manifest_sha256'] and Path(e['scene']).name == f'{scene}.yml', (scene, e['scene'])
    queries = []
    for item in manifest['suites']:
        task = item['task']; z = np.load(ROOT/'experiments/cabinet'/tests/'suites'/item['file'], allow_pickle=True)
        plans = {}
        for src, (run, label) in LABEL.items():
            label = label.format(t=task)
            with np.load(paths[run]/(label+'.npz'), allow_pickle=True) as zz: traj = list(zz['trajectories']); ids = [int(x) for x in zz['query_ids']] if 'query_ids' in zz.files else list(range(len(traj)))   # release MPPI saves queries in order
            plans[src] = (traj, ids, json.loads((paths[run]/(label+'.score.json')).read_text())[label]['per_query'])
        ok = {s: np.array([plans[s][2][str(i)]['success'] for i in range(len(plans[s][1]))]) for s in ('pnt_ntrl', 'ntrl_ntrl')}
        picks = []
        for want in [(True, False), (False, True), (True, True), (False, False)]:
            ids = np.flatnonzero((ok['pnt_ntrl'] == want[0]) & (ok['ntrl_ntrl'] == want[1]))
            picks += [int(i) for i in ids[:1]]
        for i in range(len(ok['pnt_ntrl'])):   # fill to PER_SUITE when a combination does not occur
            if len(picks) >= PER_SUITE: break
            if i not in picks: picks.append(i)
        for qi in sorted(picks):
            entry = dict(suite=task, query=qi, start_region=str(z['start_region'][qi]), goal_region=str(z['goal_region'][qi]), plans={})
            sources = {'witness': (z['witness_trajectories'][qi], dict(success=True, reasons=[], n_waypoints=len(z['witness_trajectories'][qi])))}
            for src, (traj, ids, score) in plans.items():
                k = ids.index(qi); sources[src] = (traj[k], score[str(qi)])
            for src, (path, verdict) in sources.items():
                path = np.asarray(path, np.float64); env, slf = judge(pc, path)
                keep = thin(len(path), [int(np.argmin(env))])
                entry['plans'][src] = dict(path=path[keep].round(5).tolist(), clearance=np.minimum(segment_min(env, keep), 1).round(4).tolist(),
                                           collide=(segment_min(env, keep) < OBS_MARGIN).astype(int).tolist(), self_collide=(segment_min(slf, keep) < SELF_MARGIN).astype(int).tolist(),
                                           success=bool(verdict['success']), reasons=verdict['reasons'], waypoints=int(verdict['n_waypoints']), min_clearance=float(env.min()))
            queries.append(entry)
        print(scene, task, picks, flush=True)
    doc = yaml.safe_load((ROOT/'experiments/cabinet/scenes'/f'{scene}.yml').read_text())
    cuboids = []
    for c in doc['cuboids']:
        assert c['pose'][3:] == [1., 0., 0., 0.], c['name']
        cuboids.append(dict(name=c['name'], lo=np.subtract(c['pose'][:3], np.divide(c['dims'], 2)).tolist(), hi=np.add(c['pose'][:3], np.divide(c['dims'], 2)).tolist(),
                            color=c.get('color', [128, 128, 128])))
    totals = {}
    for src, (run, method) in TOTALS.items():
        t = evaluation[run]['totals'][method]; totals[src] = dict(success=t['success'], n=t['n'], per_suite=t['per_suite'])
    return dict(scene=scene, tests=tests, runs=dict(pntfields=runs[0], ntrl_demo=runs[1]), cuboids=cuboids, totals=totals, queries=queries)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scenes', nargs='+', default=list(SCENES))
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('cabinet.json')); a = p.parse_args()
    out = dict(source='experiments/pntfields_cabinet/visualizer/export_cabinet.py', robot=str(R.URDF.relative_to(ROOT)), joint_names=R.JOINTS,
               joints=urdf_joints(R.URDF, R.JOINTS), base='base_link', ee_link=R.EE_LINK, meshes=robot_meshes(), sources=SOURCES, scenes=[])
    for scene in a.scenes:
        tests, *runs = SCENES[scene]
        pc = PieceClearance(ROOT/'experiments/cabinet/scenes'/f'{scene}.yml', 'split', 'cpu')
        out['scenes'].append(export_scene(scene, tests, runs, pc))
    rng = np.random.default_rng(5); ref_q = rng.uniform(-math.pi, math.pi, (16, 6)); links = sorted({m['link'] for m in out['meshes']} | {R.EE_LINK})
    with torch.no_grad(): t = R.Kinematics().transforms(torch.tensor(ref_q, dtype=torch.float64), links)
    out['reference'] = dict(q=ref_q.round(6).tolist(), links=links, poses=torch.stack([t[l] for l in links], 1).reshape(16, -1, 16).numpy().round(7).tolist())
    a.output.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {a.output} ({a.output.stat().st_size/1e6:.2f} MB): {len(out["meshes"])} meshes, {sum(len(s["queries"]) for s in out["scenes"])} queries')


if __name__ == '__main__':
    main()
