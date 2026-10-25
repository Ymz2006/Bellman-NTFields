"""Score trajectories from ANY planner under one common validator + path metrics.

Acceptance (same for every planner): endpoints match the query, joint limits,
obstacle clearance >= 0.005 m, self clearance >= 0.002 m, and the zone tilt gate
(<=20 deg, OR-logic inside the overlap band) enforced at every waypoint after
densifying to a maximum per-joint step of 0.001 rad.

Reports strict success plus BOTH path lengths: end-effector path in metres and
joint-space path in radians.
"""
import argparse, json
import numpy as np, torch, yaml
from curobo.geom.types import WorldConfig
import ur5e_plan_eval_multi as E
from collision_utils import (load_cuboid_geometry, build_self_collision_mask,
                             sphere_cuboid_distance, sphere_sphere_distance)

p = argparse.ArgumentParser()
p.add_argument('--suite', required=True)
p.add_argument('--trajectories', required=True, nargs='+',
               help='one or more npz files; each must hold an object array of (T,6) rad paths')
p.add_argument('--labels', nargs='+')
p.add_argument('--scene', default='/workspace/scene_ur5e_tabletop_noobs_ov10.yml')
p.add_argument('--robot', default='/workspace/ur5e_arm.yml')
p.add_argument('--norm_file', default='/workspace/datasets/arm/UR5e_noobs_ov10_corner020/joint_normalization.npz')
p.add_argument('--step', type=float, default=0.001)
p.add_argument('--tilt_deg', type=float, default=20.0)
p.add_argument('--obs_margin', type=float, default=0.005)
p.add_argument('--self_margin', type=float, default=0.002)
p.add_argument('--json_out', default=None)
a = p.parse_args()
dev = 'cuda:0'

scene_yaml = yaml.safe_load(open(a.scene))
scene = WorldConfig.from_dict(scene_yaml)
rw, rm = E.setup_robot_world(a.robot, scene, dev)
task_axes = E.load_task_axes(a.scene)
cc, ch = load_cuboid_geometry(a.scene, dev)
rcfg = yaml.safe_load(open(a.robot))
mask, _ = build_self_collision_mask(
    E.SPHERE_YML_PATH, rcfg['robot_cfg']['kinematics'].get('self_collision_ignore', {}),
    E.COLLISION_LINK_NAMES, device=dev)
z = np.load(a.suite, allow_pickle=True)
zones = json.loads(str(z['corner_zones_json']))
starts, goals = z['starts_rad'], z['goals_rad']
nz = np.load(a.norm_file)
q_lo = nz['q_min'] if 'q_min' in nz.files else nz['q_center'] - nz['q_range']
q_hi = nz['q_max'] if 'q_max' in nz.files else nz['q_center'] + nz['q_range']


def densify(q, step):
    out = [q[0]]
    for i in range(1, len(q)):
        d = np.abs(q[i] - q[i - 1]).max()
        n = max(1, int(np.ceil(d / step)))
        for k in range(1, n + 1):
            out.append(q[i - 1] + (q[i] - q[i - 1]) * (k / n))
    return np.asarray(out, dtype=np.float32)


@torch.no_grad()
def score(q_rad, i):
    r = dict(planner=q_rad is not None and len(q_rad) >= 2)
    if not r['planner']:
        return dict(success=False, reasons=['no_path'], ee_len=np.nan, cs_len=np.nan)
    q = np.asarray(q_rad, dtype=np.float32)
    r['endpoint'] = bool(np.allclose(q[0], starts[i], atol=1e-4, rtol=0)
                         and np.allclose(q[-1], goals[i], atol=1e-4, rtol=0))
    d = densify(q, a.step)
    r['joint_limits'] = bool(((d >= q_lo - 1e-5) & (d <= q_hi + 1e-5)).all())
    qt = torch.as_tensor(d, device=dev)
    ee_len = cs_len = 0.0
    ok_coll = ok_tilt = True
    for s in range(0, len(d), 20000):
        chunk = qt[s:s + 20000]
        st = rm.get_state(chunk)
        sph = st.link_spheres_tensor
        c, rad = sph[:, :, :3], sph[:, :, 3]
        if float(sphere_cuboid_distance(c, rad, cc, ch).min()) < a.obs_margin:
            ok_coll = False
        if float(sphere_sphere_distance(c, rad, mask).min()) < a.self_margin:
            ok_coll = False
        cup, pour, _, pos = E._zone_tilts_batch(chunk, rm, return_pos=True, task_axes=task_axes)
        cup_m, pour_m, ov_m = E.corner_masks_torch(pos[:, :2], zones, assign_outside=True)
        # OR logic inside the overlap band: either task's gate may be satisfied.
        good = torch.where(ov_m, (cup <= a.tilt_deg) | (pour <= a.tilt_deg),
                           torch.where(cup_m, cup <= a.tilt_deg, pour <= a.tilt_deg))
        if not bool(good.all()):
            ok_tilt = False
    dd = np.diff(d, axis=0)
    cs_len = float(np.linalg.norm(dd, axis=1).sum())
    pos_all = []
    for s in range(0, len(d), 20000):
        pos_all.append(rm.get_state(qt[s:s + 20000]).ee_pose.position.cpu().numpy())
    P = np.concatenate(pos_all, axis=0)
    ee_len = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    r['collision'] = ok_coll
    r['tilt'] = ok_tilt
    return dict(success=all(r.values()), reasons=[k for k, v in r.items() if not v],
                ee_len=ee_len, cs_len=cs_len, n_waypoints=int(len(q)))


labels = a.labels or [f'traj{i}' for i in range(len(a.trajectories))]
summary = {}
for label, path in zip(labels, a.trajectories):
    tz = np.load(path, allow_pickle=True)
    key = 'trajectories' if 'trajectories' in tz.files else tz.files[0]
    trajs = list(tz[key])
    ids = [int(x) for x in tz['query_ids']] if 'query_ids' in tz.files else list(range(len(trajs)))
    rows = [score(t, i) for t, i in zip(trajs, ids)]
    ok = [r for r in rows if r['success']]
    el = np.array([r['ee_len'] for r in ok]); cl = np.array([r['cs_len'] for r in ok])
    summary[label] = dict(
        n=len(rows), success=len(ok), rate=len(ok) / max(1, len(rows)),
        ee_len_mean=float(el.mean()) if len(ok) else float('nan'),
        cs_len_mean=float(cl.mean()) if len(ok) else float('nan'),
        solved_ids=[i for i, r in zip(ids, rows) if r['success']],
        per_query={int(i): r for i, r in zip(ids, rows)})
    pt = z['pair_type'] if 'pair_type' in z.files else None
    if pt is not None:
        for t, nm in [(0, 'cupping-only'), (1, 'pouring-only'), (2, 'cross')]:
            sel = [j for j, i in enumerate(ids) if int(pt[i]) == t]
            if not sel:
                continue
            g = [rows[j] for j in sel]
            gok = [r for r in g if r['success']]
            gl = np.array([r['ee_len'] for r in gok]); gc = np.array([r['cs_len'] for r in gok])
            summary.setdefault(label + '_groups', {})[nm] = dict(
                n=len(g), success=len(gok),
                ee_len_mean=float(gl.mean()) if len(gok) else float('nan'),
                cs_len_mean=float(gc.mean()) if len(gok) else float('nan'))
    print(f'{label:28s} strict {len(ok):3d}/{len(rows):3d}  '
          f'EE path {summary[label]["ee_len_mean"]:.3f} m  '
          f'C-space path {summary[label]["cs_len_mean"]:.3f} rad')
    from collections import Counter
    cnt = Counter(x for r in rows if not r['success'] for x in r['reasons'])
    if cnt:
        print(f'{"":28s} failures: {dict(cnt)}')
for label in list(summary):
    if label.endswith('_groups'):
        for nm, g in summary[label].items():
            print(f'    {label[:-7]:18s} {nm:14s} {g["success"]:3d}/{g["n"]:3d}  '
                  f'EE {g["ee_len_mean"]:.3f} m  C-space {g["cs_len_mean"]:.3f} rad')
if a.json_out:
    json.dump(summary, open(a.json_out, 'w'), indent=1, default=str)
    print('wrote', a.json_out)
