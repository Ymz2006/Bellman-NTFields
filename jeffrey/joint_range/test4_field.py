"""Time-field data for the Test 4 explorer (host, read-only): learned ntrl-demo travel time and speed against ground truth
on the lines through a picked grid pose, for one scene (cabinet or cabinet_strong) and the Test 4 checkpoints trained in
it. The page is test4_template.html, built by build_test4_viewer.py; timefield_check.js checks its JavaScript on this output.

Same construction as timefield.py + timefield_lines.py (the cabinet time-field explorer), generalized to a scene:
- Grid: the scene's sampling box at 2.5 cm; one IK pose per point, spread from the same seed pose (frozen cabinet test query
  same_compartment start 87) with the same orientation (its approach axis turned to +x), so both scenes use one IK branch;
  then `relax` towards clearance in this scene. Label: the release clearance label in pure torch with this scene's cuboids,
  checked against the cuRobo labels of the scene's Test 4 data run.
- Ground truth: target speed s' from the label at every pose and at 16 steps along every edge to the 98 neighbours of the
  5 x 5 x 5 block; edge cost = integral of |dx| / s'; true time = Dijkstra (run in the page). Lines: the label at the 15
  interior points of every +x, +y, +z edge (lines.bin). `tightness`: the 6-D path optimizer's gain over graph paths.
- Learned: each checkpoint's effective weights (the page runs the network), validation values from PyTorch.
- Arm: the release collision spheres (experiments/face_speed/robot.py, padded radii) in the frame of the nearest link on the
  page's kinematic chain (the gripper's finger links hang off robotiq_arg2f_base_link through locked joints), so the page can
  draw the whole arm; validation: world sphere centres of three poses.

  CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/test4_field.py --scene cabinet_strong --output jeffrey/joint_range/test4_viewer/cabinet_strong
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import timefield as TF
import timefield_lines as TL
from timefield import R, ROOT, H, JUMP, STEPS, ALPHA, MARGIN, OFFSET, SELF_SCALE, ARM, DIRS, N26, target_speed

SCENES = {
    'cabinet': dict(data='registered_20261101T021848Z-afa3dd8b', lift=(0., 0., 0.),
                    models={'collide': ('Collision samples (1/3 of pairs)', 'registered_20261101T022230Z-061c5d00', 'trained normally'),
                            'mixed': ('Mixed pairs only (control)', 'registered_20261101T022230Z-b9f62b88', 'trained normally')}),
    'cabinet_strong': dict(data='registered_20261101T022101Z-c171d565', lift=(0., .08, .16),
                           models={'collide': ('Collision samples (1/3 of pairs)', 'registered_20261101T023339Z-856a30d0', 'collapsed: slow everywhere'),
                                   'mixed': ('Mixed pairs only (control, seed 1)', 'registered_20261101T030834Z-bc235281', 'trained normally')}),
}
# timefield.py's presets (cabinet); in the strong cabinet the middle and top compartments sit 0.08 and 0.16 m higher
BASE_PRESETS = {'middle compartment': ((0.705, 0.330, 0.315), 1), 'top compartment': ((0.705, 0.330, 0.530), 2), 'bottom compartment': ((0.705, 0.330, 0.090), 0),
                'in front of the cupboard': ((0.405, 0.330, 0.315), 1), 'beside the cupboard': ((0.405, 0.080, 0.165), 0)}


class Label(TF.Label):
    """timefield.Label (the release clearance label in pure torch) with the cuboids of a given scene."""
    def __init__(self, dev, scene):
        self.robot = R.Robot().to(dev); self.cu = R.collision_utils(); doc = yaml.safe_load(scene.read_text())
        self.cen = torch.tensor([c['pose'][:3] for c in doc['cuboids']], dtype=torch.float32, device=dev)
        self.half = torch.tensor([c['dims'] for c in doc['cuboids']], dtype=torch.float32, device=dev)/2


def arm_spheres(robot, chain_links):
    """Padded collision spheres in the frame of the nearest link on the page's chain: [link, x, y, z, radius, ignored]."""
    kin = robot.kinematics; q0 = torch.zeros(1, 6); out = []
    for k, link in enumerate(robot.collision_links):
        anc = link
        while anc not in chain_links: anc = kin.joint_of[anc][1]
        t = kin.transforms(q0, [link, anc]); rel = torch.linalg.inv(t[anc][0])@t[link][0]
        for i in torch.nonzero(robot.sphere_link == k)[:, 0].tolist():
            c = rel[:3, :3]@robot.sphere_centers[i]+rel[:3, 3]
            out.append([anc, *[round(float(v), 6) for v in c], round(float(robot.sphere_radii[i]), 6), int(i in R.IGNORED_SPHERES)])
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', choices=list(SCENES), required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    dev = a.device; started = time.monotonic(); out = a.output; out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0); rng = np.random.default_rng(0)
    cfg = SCENES[a.scene]; scene = ROOT/f'experiments/cabinet/scenes/{a.scene}.yml'; doc = yaml.safe_load(scene.read_text())
    norm = np.load(R.NORMALIZATION); q_range = norm['q_range'].astype(np.float64); q_min, q_max = norm['q_min'].astype(np.float64), norm['q_max'].astype(np.float64)
    label = Label(dev, scene); report = {}; qr = torch.tensor(q_range, device=dev)

    # 1. the pure-torch label against the cuRobo labels of this scene's Test 4 data (control set)
    S = ROOT/'results'/cfg['data']/'data/control'; X = np.load(S/'sampled_points.npy', mmap_mode='r'); Dc = np.load(S/'clearance.npy', mmap_mode='r')
    pick = np.sort(rng.choice(len(X), 20000, replace=False)); xs = torch.tensor(np.concatenate([X[pick, :6], X[pick, 6:]]), device=dev); ref = np.concatenate([Dc[pick, 0], Dc[pick, 1]])
    mine = label(xs*qr.float()).cpu().numpy(); dl = np.abs(mine-ref)
    report['label_check'] = dict(configs=int(len(ref)), max_abs_m=float(dl.max()), p99_abs_m=float(np.quantile(dl, .99))); print('label check', report['label_check'], flush=True)
    assert dl.max() < 1e-4, 'pure-torch label does not reproduce the stored labels'

    # 2. grid and inverse kinematics from timefield.py's seed and orientation, then relax towards clearance in this scene
    lo = np.array([doc['sampling'][k][0] for k in 'xyz']); hi = np.array([doc['sampling'][k][1] for k in 'xyz'])
    n = np.round((hi-lo)/H).astype(int)+1; ngrid = int(n.prod()); gi = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing='ij'), -1).reshape(-1, 3)
    P = lo+gi*H; flat = lambda ijk: (ijk[..., 0]*n[1]+ijk[..., 1])*n[2]+ijk[..., 2]
    suite = next(s for s in json.loads((TF.TESTS/'suite_manifest.json').read_text())['suites'] if s['task'] == TF.SEED[0])
    seed_q = np.load(TF.TESTS/'suites'/suite['file'], allow_pickle=True)[TF.SEED[1]][TF.SEED[2]].astype(np.float64)
    t_seed = R.Kinematics().transforms(torch.tensor(seed_q[None]), [R.EE_LINK])[R.EE_LINK][0].numpy()
    gx = t_seed[:3, 0].copy(); gx[0] = 0; gx /= np.linalg.norm(gx); gz = np.array([1., 0., 0.]); R0 = np.stack([gx, np.cross(gz, gx), gz], 1)
    ik = TF.IK(dev, R0); seed_node = int(flat(np.clip(np.round((t_seed[:3, 3]-lo)/H).astype(int), 0, n-1)))
    Pt = torch.tensor(P, dtype=torch.float64, device=dev); Q, solved, waves = TF.flood_fill(ik, Pt, gi, n, seed_node, seed_q, q_min, q_max)
    assert solved.sum() > 1000, 'seed IK failed'
    ids = np.full(ngrid, -1, np.int32); nodes = np.nonzero(solved)[0]; ids[nodes] = np.arange(len(nodes)); N = len(nodes)
    report['grid'] = dict(shape=n.tolist(), points=ngrid, solved=int(N), waves=waves); print('ik', report['grid'], flush=True)
    Qn = torch.tensor(Q[nodes], device=dev); pairs = []
    for d in N26:
        if next(c for c in d if c) < 0: continue
        j = gi[nodes]+np.array(d); inb = np.all((j >= 0) & (j < n), 1); dst = np.full(N, -1); dst[inb] = ids[flat(j[inb])]; m = dst >= 0; pairs.append(np.stack([np.nonzero(m)[0], dst[m]], 1))
    pairs = np.concatenate(pairs); d_before = label(Qn.float()).cpu().numpy()
    Qn = TF.relax(ik, label, Qn, Pt[nodes], pairs, q_min, q_max); Q[nodes] = Qn.cpu().numpy(); d_after = label(Qn.float()).cpu().numpy(); pe, Re, _ = ik.fk(Qn)
    report['relax'] = dict(free_before=float((d_before > 0).mean()), free_after=float((d_after > 0).mean()), neighbour_jump_max_rad=float(np.abs(Q[nodes[pairs[:, 0]]]-Q[nodes[pairs[:, 1]]]).max()),
                           approach_x_median=float(Re[:, 0, 2].median()))
    report['ik_error'] = dict(position_max_m=float((pe-Pt[nodes]).norm(dim=1).max())); print('relax', report['relax'], report['ik_error'], flush=True)

    # 3. ground-truth speed at the nodes
    d_node = label(Qn.float()); s_node = target_speed(d_node); dn = d_node.cpu().numpy()
    panels = [c for c in doc['cuboids'] if c['name'].startswith('shelf_')]
    cup_lo = np.min([np.subtract(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0); cup_hi = np.max([np.add(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0)
    in_cup = np.all((P[nodes] >= cup_lo) & (P[nodes] <= cup_hi), 1)
    report['grid'].update(free_outside_cupboard=float((dn[~in_cup] > 0).mean()), free_inside_cupboard=float((dn[in_cup] > 0).mean()), cupboard_box=[cup_lo.tolist(), cup_hi.tolist()])
    print('grid', report['grid'], flush=True)

    # 4. edges: 49 canonical directions of the 5x5x5 block, cost = int |dx| / s' over the straight joint-space segment
    edges = np.full((N, len(DIRS)), np.inf, np.float32); t = torch.linspace(0, 1, STEPS+1, dtype=torch.float64, device=dev); dropped = 0
    for k, d in enumerate(DIRS):
        j = gi[nodes]+np.array(d); inb = np.all((j >= 0) & (j < n), 1); src = np.nonzero(inb)[0]; dst = ids[flat(j[inb])]; keep = dst >= 0; src, dst = src[keep], dst[keep]
        jump = np.abs(Q[nodes[dst]]-Q[nodes[src]]).max(1) > JUMP; dropped += int(jump.sum()); src, dst = src[~jump], dst[~jump]
        for s in range(0, len(src), 4000):
            a0, a1 = Qn[src[s:s+4000]], Qn[dst[s:s+4000]]; seg = a0[:, None]+t[None, :, None]*(a1-a0)[:, None]
            inv = 1/target_speed(label(seg.reshape(-1, 6))).reshape(len(a0), -1); edges[src[s:s+4000], k] = (((a1-a0)/qr).norm(dim=1)*torch.trapezoid(inv, t, dim=1)).float().cpu().numpy()
    report['edges'] = dict(directions=len(DIRS), finite=int(np.isfinite(edges).sum()), dropped_jumps=dropped); print('edges', report['edges'], flush=True)

    # 5. files the line step reads back as stored (float32), as timefield_lines.py does
    nodes_blob = np.concatenate([gi[nodes].astype(np.float32), Q[nodes].astype(np.float32), dn[:, None].astype(np.float32), s_node.float().cpu().numpy()[:, None]], 1).astype(np.float32)
    (out/'nodes.bin').write_bytes(nodes_blob.tobytes()); (out/'edges.bin').write_bytes(edges.tobytes())
    A = np.frombuffer((out/'nodes.bin').read_bytes(), np.float32).reshape(N, 11); Qs = A[:, 3:9].astype(np.float64); Ds = A[:, 9].astype(np.float64)
    G2N = np.full(ngrid, -1); G2N[flat(gi[nodes])] = np.arange(N); SUB = TL.SUB
    LD = np.full((N, 3, SUB-1), np.nan, np.float32); ts = torch.linspace(0, 1, SUB+1, dtype=torch.float64, device=dev); cost_err, stray = [], []
    for axis in range(3):
        d = np.zeros(3, int); d[axis] = 1; kdir = DIRS.index(tuple(d)) if tuple(d) in DIRS else [list(x) for x in DIRS].index(d.tolist())
        j = gi[nodes]+d; inb = np.all(j < n, 1); src = np.nonzero(inb)[0]; dst = G2N[flat(j[inb])]; keep = dst >= 0; src, dst = src[keep], dst[keep]
        for s in range(0, len(src), 4000):
            sa, sb = src[s:s+4000], dst[s:s+4000]; a0 = torch.tensor(Qs[sa], device=dev); a1 = torch.tensor(Qs[sb], device=dev); seg = a0[:, None]+ts[None, :, None]*(a1-a0)[:, None]
            dd = label(seg[:, 1:SUB].reshape(-1, 6)).reshape(len(sa), SUB-1).double(); LD[sa, axis] = dd.float().cpu().numpy()
            full = torch.cat([torch.tensor(Ds[sa], device=dev)[:, None], dd, torch.tensor(Ds[sb], device=dev)[:, None]], 1)
            cost = ((a1-a0)/qr).norm(dim=1)*torch.trapezoid(1/target_speed(full), ts, dim=1); ref = torch.tensor(edges[sa, kdir], device=dev).double()
            cost_err.append(((cost-ref).abs()/ref).cpu().numpy())
            pe_ = ik.fk(seg[:, 1:SUB].reshape(-1, 6))[0].reshape(len(sa), SUB-1, 3); nominal = torch.tensor(lo+gi[nodes][sa]*H, device=dev)[:, None]+ts[None, 1:SUB, None]*torch.tensor(d*H, device=dev, dtype=torch.float64)
            off = pe_-nominal; stray.append(torch.sqrt((off**2).sum(-1)-off[..., axis]**2).amax(1).cpu().numpy())
    ce, st = np.concatenate(cost_err), np.concatenate(stray)
    lrep = dict(cost_vs_edges_rel_max=float(ce.max()), cost_vs_edges_rel_p99=float(np.quantile(ce, .99)), gripper_off_line_max_m=float(st.max()), gripper_off_line_p99_m=float(np.quantile(st, .99)))
    print('lines', lrep, flush=True); assert ce.max() < 1e-3; (out/'lines.bin').write_bytes(LD.tobytes())

    # 6. presets: nearest grid pose with >= 5 mm clearance to timefield.py's points, compartment heights lifted in this scene
    free_ids = np.nonzero(Ds > OFFSET)[0]; preset_nodes = {}; preset_report = {}
    for name, (xyz, comp) in BASE_PRESETS.items():
        target = np.array(xyz)+np.array([0, 0, cfg['lift'][comp]]); c = (target-lo)/H
        k = int(free_ids[np.argmin(np.abs(gi[nodes[free_ids]]-c).sum(1))]); preset_nodes[name] = k
        preset_report[name] = dict(target=target.round(4).tolist(), node=k, position=(lo+gi[nodes[k]]*H).round(4).tolist(), clearance_m=float(Ds[k]))
    report['presets'] = preset_report; print('presets', json.dumps(preset_report), flush=True)

    # 7. tightness of the graph truth: the 6-D optimizer's gain over graph paths from two starts
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    E = np.frombuffer((out/'edges.bin').read_bytes(), np.float32).reshape(N, len(DIRS))
    src, kk = np.nonzero(np.isfinite(E)); dst = G2N[flat(gi[nodes][src]+np.array(DIRS)[kk])]; w = E[src, kk].astype(np.float64)
    G = coo_matrix((np.concatenate([w, w]), (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(N, N)).tocsr(); report['tightness'] = {}
    for name in ('in front of the cupboard', 'middle compartment'):
        s0 = preset_nodes[name]; dist, pred = dijkstra(G, indices=s0, return_predecessors=True); ok = np.isfinite(dist); ok[s0] = False
        groups = {'outside, collision-free': ok & ~in_cup & (Ds > OFFSET), 'cupboard, collision-free': ok & in_cup & (Ds > OFFSET), 'in contact': ok & (Ds <= 0)}
        report['tightness'][name] = {}
        for gname, m in groups.items():
            if m.sum() < 10: report['tightness'][name][gname] = None; continue
            goals = rng.choice(np.nonzero(m)[0], min(96, int(m.sum())), replace=False); paths = []
            for g in goals:
                chain = [g]
                while chain[-1] != s0: chain.append(pred[chain[-1]])
                paths.append(torch.tensor(Qs[chain[::-1]], device=dev))
            _, after = TF.optimize_paths(label, paths, qr); gain = 1-after.double().cpu().numpy()/dist[goals]
            report['tightness'][name][gname] = dict(paths=int(len(goals)), graph_time_median=float(np.median(dist[goals])), optimizer_gain_median=float(np.median(gain)), optimizer_gain_p90=float(np.quantile(gain, .9)))
        if name == 'middle compartment': ref_dist = dist
    print('tightness', json.dumps(report['tightness']), flush=True)

    # 8. learned fields: effective weights and validation pairs (timefield.py), then learned values at line points
    FIELD = TF.load_field_module(); xg = torch.tensor(Qs/q_range, device=dev).float(); checkpoints = {}
    val_pairs = np.stack([rng.integers(0, N, 48), rng.integers(0, N, 48)], 1); val_pairs[:8, 0] = preset_nodes['middle compartment']
    s0 = preset_nodes['middle compartment']; lval = dict(source=int(s0), lines=[], learned={}); picks = []
    for axis in range(3):
        d = np.zeros(3, int); d[axis] = 1; kdir = [list(x) for x in DIRS].index(d.tolist()); pts = TL.line_points(dict(grid=dict(shape=n.tolist())), gi[nodes][s0], axis, G2N, LD)
        Tl, Sl = TL.line_truth(dict(offset=OFFSET, margin=MARGIN, alpha=ALPHA), pts, axis, ref_dist, Qs, Ds, LD, E, kdir, q_range)
        lval['lines'].append(dict(axis=axis, count=len(pts), T=[float(v) if np.isfinite(v) else -1. for v in Tl], S=Sl.tolist()))
        for q_ in rng.choice(len(pts), 12, replace=False):
            a_, b_, k, _, _ = pts[q_]; picks.append((axis, int(q_), Qs[a_]/q_range if b_ < 0 else (Qs[a_]+(k/SUB)*(Qs[b_]-Qs[a_]))/q_range))
    xl = torch.tensor(np.stack([np.concatenate([Qs[s0]/q_range, x]) for _, _, x in picks]), dtype=torch.float32, device=dev)
    for key, (title, run, status) in cfg['models'].items():
        net = FIELD.load_metric(ROOT/'results'/run/'training/checkpoint.pt', dev, 'metric', 'release').net
        x01 = torch.cat([xg[val_pairs[:, 0]], xg[val_pairs[:, 1]]], 1)
        with torch.enable_grad():
            tau, _, xp = net.out(x01); grad = torch.autograd.grad(tau.sum(), xp)[0]
            taul, _, xpl = net.out(xl); gradl = torch.autograd.grad(taul.sum(), xpl)[0]
        with torch.no_grad(): eb = TF.embed(net, xg[val_pairs[0, 1]][None])[0, :8]
        blob = TF.weights_blob(net); (out/f'net_{key}.bin').write_bytes(blob.tobytes())
        checkpoints[key] = dict(title=title, run=run, status=status, weights=int(len(blob)),
                                validation=dict(T=tau[:, 0].tolist(), speed1=(1/grad[:, 6:].norm(dim=1)).tolist(), speed0=(1/grad[:, :6].norm(dim=1)).tolist(), e_head=eb.tolist()))
        lval['learned'][key] = dict(points=[[ax, q_] for ax, q_, _ in picks], T=taul[:, 0].tolist(), S=(1/gradl[:, 6:].norm(dim=1)).tolist())
        print('checkpoint', key, run, 'T median of validation pairs', float(tau[:, 0].median()), flush=True); del net; torch.cuda.empty_cache()

    # 9. the arm's collision spheres for the page, and world centres of three poses to check the page's placement
    chain = TF.ET_chain(); chain_links = {'base_link'}|{c['child'] for c in chain}; robot = R.Robot()
    spheres = arm_spheres(robot, chain_links); check_nodes = [int(s0), int(preset_nodes['in front of the cupboard']), int(rng.integers(N))]
    world = robot.link_spheres(torch.tensor(Qs[check_nodes], dtype=torch.float32)).numpy()[:, :, :3]

    meta = dict(scene=a.scene, grid=dict(lo=lo.tolist(), h=H, shape=n.tolist()), nodes=int(N), dirs=[list(x) for x in DIRS], orientation=R0.tolist(), seed=dict(suite=TF.SEED[0], key=TF.SEED[1], index=TF.SEED[2], node=int(ids[seed_node])),
                presets=preset_nodes, q_range=q_range.tolist(), alpha=ALPHA, margin=MARGIN, offset=OFFSET, steps=STEPS, jump=JUMP, cupboard=[cup_lo.tolist(), cup_hi.tolist()],
                boxes=[dict(name=c['name'], center=c['pose'][:3], dims=c['dims']) for c in doc['cuboids']], sampling={k: doc['sampling'][k] for k in 'xyz'}, compartments=doc['compartments'],
                chain=chain, arm_links=ARM, arm_spheres=spheres, ignored_spheres=list(R.IGNORED_SPHERES), checkpoints=checkpoints,
                validation=dict(pairs=val_pairs.tolist(), dijkstra_start=int(s0), dijkstra_sample=[[int(i), float(ref_dist[i]) if np.isfinite(ref_dist[i]) else -1.] for i in rng.choice(N, 400, replace=False)],
                                spheres=dict(nodes=check_nodes, world=world.round(7).tolist())),
                report=report, elapsed_seconds=time.monotonic()-started)
    (out/'meta.json').write_text(json.dumps(meta, separators=(',', ':')))
    (out/'lines.json').write_text(json.dumps(dict(sub=SUB, report=lrep, validation=lval), separators=(',', ':')))
    print('done', a.scene, N, 'nodes', round(time.monotonic()-started), 's', flush=True)


if __name__ == '__main__': main()
