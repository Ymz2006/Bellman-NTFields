"""Export the arm's collision geometry for the browser viewer (experiments/face_speed/visualizer).

Everything comes from the same code the experiment uses (experiments/face_speed/robot.py, clearance.py):
  - URDF kinematic tree (joint origins, moving joint index); the viewer redoes the forward kinematics,
  - the release collision spheres with cuRobo's padded radii, their links, the ignored shoulder spheres,
  - the release self-collision mask as the list of checked sphere pairs,
  - the box cover (one box per rigid body in its reference link frame) from which the 30 faces follow,
  - the collision meshes of every link with their URDF collision origin and scale,
  - the cabinet scene cuboids, and for a few frozen queries per suite the certified witness path plus the
    Bellman RK4 paths of the release-speed and box-face-speed fields with their scorer verdicts,
  - reference per-piece clearances for 48 configurations (exact, from clearance.py) so the viewer's
    JavaScript geometry can be checked against the experiment (check.mjs).

Read-only; writes one JSON file.
  ~/.venvs/bellman-ntfield/bin/python experiments/face_speed/visualizer/export.py
"""
import argparse, base64, json, math, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import trimesh
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.face_speed import robot as R
from experiments.face_speed.clearance import PieceClearance, MARGIN, OFFSET, SELF_SCALE

SCENE = ROOT/'experiments/cabinet/scenes/cabinet.yml'
TESTS = ROOT/'experiments/cabinet/tests'
RUNS = {'release': 'registered_20261025T070429Z-b9a2383c', 'faces': 'registered_20261025T070501Z-aaa42561'}
PER_SUITE = 8


def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def origin_matrix(el):
    xyz = [float(x) for x in (el.get('xyz', '0 0 0') if el is not None else '0 0 0').split()]
    rpy = [float(x) for x in (el.get('rpy', '0 0 0') if el is not None else '0 0 0').split()]
    t = np.eye(4); t[:3, :3] = R.rpy_matrix(*rpy).numpy(); t[:3, 3] = xyz; return t


def thin(path, n=160):
    path = np.asarray(path, np.float32)
    if len(path) <= n: return path
    keep = np.unique(np.linspace(0, len(path)-1, n).round().astype(int)); return path[keep]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path(__file__).with_name('geometry.json')); a = p.parse_args()
    robot = R.Robot(); boxes = R.BoxCover(robot)
    root = ET.parse(R.URDF).getroot()
    joints = []
    for j in root.findall('joint'):
        name = j.get('name')
        joints.append(dict(name=name, parent=j.find('parent').get('link'), child=j.find('child').get('link'),
                           origin=origin_matrix(j.find('origin')).reshape(-1).round(9).tolist(), index=R.JOINTS.index(name) if name in R.JOINTS else -1))
    meshes = []
    for link in root.findall('link'):
        col = link.find('collision')
        if col is None: continue
        mesh = col.find('geometry/mesh')
        if mesh is None: continue
        m = trimesh.load(R.URDF.parent/mesh.get('filename'), force='mesh')
        scale = [float(x) for x in mesh.get('scale', '1 1 1').split()]
        m.apply_scale(scale); m.merge_vertices()
        meshes.append(dict(link=link.get('name'), file=mesh.get('filename'), origin=origin_matrix(col.find('origin')).reshape(-1).round(9).tolist(),
                           positions=b64(m.vertices.astype(np.float32)), indices=b64(m.faces.astype(np.uint32)), triangles=int(len(m.faces))))
    S = len(robot.sphere_radii)
    mask = robot.self_mask.cpu().numpy()
    pairs = [[i, j] for i in range(S) for j in range(i+1, S) if not mask[i, j]]
    spheres = [dict(name=robot.sphere_names[i], link=int(robot.sphere_link[i]), center=robot.sphere_centers[i].tolist(),
                    radius=float(robot.sphere_radii[i]), ignored=i in R.IGNORED_SPHERES) for i in range(S)]
    bodies = [dict(name=b, reference=boxes.reference[k], links=R.BODIES[b][1], center=boxes.box_center[k].tolist(), half=boxes.box_half[k].tolist())
              for k, b in enumerate(boxes.body_names)]
    import yaml
    doc = yaml.safe_load(SCENE.read_text())
    cuboids = [dict(name=c['name'], lo=(np.subtract(c['pose'][:3], np.divide(c['dims'], 2))).tolist(), hi=(np.add(c['pose'][:3], np.divide(c['dims'], 2))).tolist(),
                    color=c.get('color', [128, 128, 128])) for c in doc['cuboids']]
    # Queries: witness path + Bellman RK4 paths of both fields.
    manifest = json.loads((TESTS/'suite_manifest.json').read_text()); queries = []
    for item in manifest['suites']:
        z = np.load(TESTS/'suites'/item['file'], allow_pickle=True); suite = str(z['suite_name']).replace('cabinet_', '') if 'suite_name' in z else item['file']
        task = item['file'].replace('cabinet_', '').replace('_100.npz', '')
        plans = {}
        for field, run in RUNS.items():
            e = ROOT/'results'/run/'evaluation'; label = f'bellman_rk4_{task}'
            with np.load(e/(label+'.npz'), allow_pickle=True) as zz: plans[field] = (list(zz['trajectories']), [int(x) for x in zz['query_ids']])
            plans[field] += (json.loads((e/(label+'.score.json')).read_text())[label]['per_query'],)
        for qi in range(PER_SUITE):
            entry = dict(suite=task, query=qi, start_region=str(z['start_region'][qi]), goal_region=str(z['goal_region'][qi]),
                         witness=thin(z['witness_trajectories'][qi]).round(5).tolist(), plans={})
            for field, (paths, ids, score) in plans.items():
                k = ids.index(qi); verdict = score[str(qi)]
                entry['plans'][field] = dict(path=thin(paths[k]).round(5).tolist(), success=bool(verdict['success']), reasons=verdict['reasons'], waypoints=int(verdict['n_waypoints']))
            queries.append(entry)
    # Reference clearances (radians in, per-piece signed clearance out) for the JS check.
    rng = np.random.default_rng(7); ref_q = []
    for qd in queries[::2][:24]: ref_q.append(qd['witness'][len(qd['witness'])//2])
    lo_lim, hi_lim = np.array([-math.pi]*6), np.array([math.pi]*6)
    ref_q += rng.uniform(lo_lim, hi_lim, (24, 6)).round(5).tolist()
    ref_q = np.asarray(ref_q, np.float32)
    reference = dict(q=ref_q.tolist())
    with torch.no_grad():
        for cover in ['spheres', 'links', 'faces']:
            pc = PieceClearance(SCENE, cover, 'cpu'); qn = (torch.tensor(ref_q)-pc.robot.q_center)/pc.robot.q_range
            reference[cover] = dict(names=pc.names, pieces=pc.pieces(qn).numpy().round(7).tolist())
        t = robot.transforms(torch.tensor(ref_q), [R.EE_LINK]); reference['ee'] = t[R.EE_LINK][:, :3, 3].numpy().round(7).tolist()
    out = dict(source='experiments/face_speed/visualizer/export.py', scene=str(SCENE.relative_to(ROOT)), urdf=str(R.URDF.relative_to(ROOT)),
               constants=dict(margin=MARGIN, offset=OFFSET, self_scale=SELF_SCALE, far=.15, sphere_buffer=robot.sphere_buffer),
               joint_names=R.JOINTS, q_min=robot.q_min.tolist(), q_max=robot.q_max.tolist(), ee_link=R.EE_LINK,
               joints=joints, collision_links=robot.collision_links, spheres=spheres, self_pairs=pairs, bodies=bodies, meshes=meshes,
               cuboids=cuboids, queries=queries, runs=RUNS, reference=reference)
    a.output.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {a.output} ({a.output.stat().st_size/1e6:.2f} MB): {len(spheres)} spheres, {len(pairs)} self pairs, {len(bodies)} boxes, '
          f'{len(meshes)} meshes ({sum(m["triangles"] for m in meshes)} triangles), {len(cuboids)} cuboids, {len(queries)} queries')


if __name__ == '__main__': main()
