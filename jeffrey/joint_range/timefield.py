"""Time-field explorer data (host, read-only): learned ntrl-demo travel time against ground truth, from a start pose to
a grid of gripper positions, for jeffrey/joint_range/timefield_template.html (built by build_timefield_viewer.py).

Grid. The scene's sampling box (the test area) at 2.5 cm, 33 x 25 x 29 points. Every point gets a joint configuration
by inverse kinematics with ONE gripper orientation, solved by damped least squares and spread outward from a seed pose
(frozen test query same_compartment, start 87) so that neighbouring points stay on its IK branch. The orientation is
the seed's with the approach axis turned exactly to +x (into the cupboard). Of 40 in-compartment test poses tried
(scratchpad orient_search.py), this branch and orientation reach the whole box with the most collision-free points.
`relax` then turns every pose towards clearance: gradient steps on min(d, 5 cm) with the gripper position held (the
orientation becomes free) and neighbouring poses kept close in joint space. A point without a solution inside the joint limits, or reachable only
by a jump of more than JUMP rad on some joint from the neighbour it was started from, stays empty.

Ground truth. The release clearance label d(q) = min(obstacle distance, 0.5 self distance) of the label code (sphere
model and distances in pure torch via experiments/face_speed/robot.py, checked here against the cuRobo labels of the
Test 3 in-cabinet pairs), the label speed S = clip(d, 5 mm, 5 cm) / 5 cm and the trainer's target
s' = 1.025 S^2 (2 - S)^2 - 0.025 (baselines/ntrl-demo/models/metric/model_train_metric.py), the speed the learned
field is fitted to. Every grid point is joined to the points of its 5 x 5 x 5 block (98 directions, the 49 canonical
ones stored) by the straight joint-space segment between the two configurations, unless a joint moves more than JUMP
rad along it. The edge cost is the travel time, the integral of |dx| / s' (x = q / q_range, the network's input
coordinates) by the trapezoid rule over 16 steps. The page's ground-truth time from a start point is the shortest
path in this graph (Dijkstra): the eikonal travel time of motions that pass through the grid's poses, up to the
grid's path discretization. It can only overestimate the unconstrained 6-D travel time. `tightness` measures by how
much: from two collision-free starts, 96 graph paths per goal group are refined by a free 6-D path optimizer
(optimize_paths) and timed densely.

Learned. For each checkpoint, the network's embedding of every grid configuration (NN.out embeds x0 and x1 separately
and returns T = 0.2 * sum over 16 groups of a soft max of |e0 - e1|), stored as float16, and the effective weights
(lip_norm applied) for the page's own forward and backward pass (field speed 1 / |dT/dx1| on the slice). `validation`
holds torch values of T and of the field speed for random pairs, to check the page's JavaScript against.

  CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/timefield.py --output jeffrey/joint_range/timefield_viewer
"""
import argparse, importlib.util, itertools, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT)]
from experiments.face_speed import robot as R

SCENE = ROOT/'experiments/cabinet/scenes/cabinet.yml'
TESTS = ROOT/'experiments/cabinet/tests'
CABINET_SET = ROOT/'results/registered_20261031T210851Z-ac3c7c7c/data/cabinet'
CHECKPOINTS = {'f100': ('Test 3 · 100 % in-cabinet', 'registered_20261031T220420Z-9105ec65'),
               'f0': ('Test 3 · 0 % in-cabinet', 'registered_20261031T211045Z-3afa500d'),
               'old': ('Old-recipe baseline', 'registered_20261025T065945Z-3c53b360')}
SEED = ('same_compartment', 'starts_rad', 87)
MIDDLE = (0.705, 0.330, 0.315)                           # the middle compartment, snapped to the grid
H, JUMP, STEPS, ALPHA = 0.025, 0.6, 16, 1.025
MARGIN, OFFSET, SELF_SCALE = .05, .005, .5
ARM = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', R.EE_LINK]
JOINT_LINKS = ARM[1:7]                                    # child link of each moving joint: its z axis is the joint axis
DIRS = [d for d in itertools.product(range(-2, 3), repeat=3) if any(d) and math.gcd(math.gcd(abs(d[0]), abs(d[1])), abs(d[2])) == 1
        and next(c for c in d if c) > 0]                  # 49 canonical directions (first nonzero component positive)
N26 = [d for d in itertools.product(range(-1, 2), repeat=3) if any(d)]


def target_speed(d):
    s = (d.clamp(OFFSET, MARGIN)/MARGIN).double()
    return ALPHA*s**2*(2-s)**2+1-ALPHA


class Label:
    """Release clearance label in pure torch: the label code's sphere-cuboid and self distances (spheres 0 and 1 ignored)."""
    def __init__(self, dev):
        self.robot = R.Robot().to(dev); self.cu = R.collision_utils(); doc = yaml.safe_load(SCENE.read_text())
        self.cen = torch.tensor([c['pose'][:3] for c in doc['cuboids']], dtype=torch.float32, device=dev)
        self.half = torch.tensor([c['dims'] for c in doc['cuboids']], dtype=torch.float32, device=dev)/2

    @torch.no_grad()
    def __call__(self, q_rad, chunk=40000):
        return torch.cat([self.differentiable(q_rad[s:s+chunk]) for s in range(0, len(q_rad), chunk)])

    def differentiable(self, q_rad):
        sp = self.robot.link_spheres(q_rad.float())
        delta = sp[:, :, None, :3]-self.cen[None, None]; clamped = (delta.abs()-self.half[None, None]).clamp_min(0)
        signed = clamped.norm(dim=-1)-sp[:, :, 3:4]
        keep = torch.ones(sp.shape[1], dtype=torch.bool, device=sp.device); keep[list(R.IGNORED_SPHERES)] = False
        obstacle = signed[:, keep].reshape(len(sp), -1).min(1).values
        self_d = self.cu.sphere_sphere_distance(sp[..., :3], sp[..., 3], self.robot.self_mask)
        return torch.minimum(obstacle, SELF_SCALE*self_d)


def optimize_paths(label, paths, qr, iters=400, K=48, sub=4, fine=16):
    """Local 6-D refinement of graph paths (lists of radian configurations): waypoints resampled to K segments by
    joint-space length, interior points moved by Adam on the travel time (sub midpoint samples per segment), resampled
    every 50 steps. Returns the travel time of the start and end path, both evaluated with `fine` samples per segment."""
    def resample(X):
        L = (X[:, 1:]-X[:, :-1]).norm(dim=2); c = torch.cat([torch.zeros_like(L[:, :1]), L.cumsum(1)], 1); u = torch.linspace(0, 1, K+1, device=X.device)*c[:, -1:]
        i = torch.searchsorted(c.contiguous(), u.contiguous()).clamp(1, X.shape[1]-1); c0, c1 = c.gather(1, i-1), c.gather(1, i); w = ((u-c0)/(c1-c0).clamp_min(1e-12))[..., None]
        return X.gather(1, (i-1)[..., None].expand(-1, -1, 6))*(1-w)+X.gather(1, i[..., None].expand(-1, -1, 6))*w
    def cost(X, m, grad):
        t = (torch.arange(m, device=X.device, dtype=X.dtype)+.5)/m; seg = X[:, :-1, None]+t[None, None, :, None]*(X[:, 1:]-X[:, :-1])[:, :, None]
        q = (seg*qr.float()).reshape(-1, 6); d = label.differentiable(q) if grad else label(q)
        inv = 1/target_speed(d).float().reshape(seg.shape[:3]); return ((X[:, 1:]-X[:, :-1]).norm(dim=2)*inv.mean(2)).sum(1)
    longest = max(len(p) for p in paths)
    X = torch.stack([torch.cat([p, p[-1:].expand(longest-len(p), 6)]) for p in paths]).float()/qr.float()
    X = resample(X); before = cost(X, fine, False)
    inner = X[:, 1:-1].clone().requires_grad_(True); opt = torch.optim.Adam([inner], lr=1e-3)
    for it in range(iters):
        Y = torch.cat([X[:, :1], inner, X[:, -1:]], 1); loss = cost(Y, sub, True).sum(); opt.zero_grad(); loss.backward(); inner.grad = torch.nan_to_num(inner.grad, 0., 0., 0.); opt.step()
        if (it+1) % 50 == 0:
            with torch.no_grad(): inner.data = resample(torch.cat([X[:, :1], inner, X[:, -1:]], 1))[:, 1:-1]
    with torch.no_grad(): after = cost(torch.cat([X[:, :1], inner, X[:, -1:]], 1), fine, False)
    return before, after


def rotvec(E):
    """Rotation vector of rotation matrices E (M, 3, 3)."""
    cos = ((E.diagonal(dim1=1, dim2=2).sum(1)-1)/2).clamp(-1, 1); angle = torch.acos(cos)
    w = torch.stack([E[:, 2, 1]-E[:, 1, 2], E[:, 0, 2]-E[:, 2, 0], E[:, 1, 0]-E[:, 0, 1]], 1)/2
    scale = torch.where(angle > 1e-6, angle/torch.sin(angle).clamp_min(1e-12), torch.ones_like(angle))
    return w*scale[:, None]


class IK:
    def __init__(self, dev, orientation):
        self.kin = R.Kinematics().to(dev); self.dev = dev
        self.R0 = torch.tensor(np.asarray(orientation), dtype=torch.float64, device=dev)

    def fk(self, q):
        t = self.kin.transforms(q, JOINT_LINKS+[R.EE_LINK]); pe = t[R.EE_LINK][:, :3, 3]; Re = t[R.EE_LINK][:, :3, :3]
        J = torch.stack([torch.cat([torch.cross(t[L][:, :3, 2], pe-t[L][:, :3, 3], dim=1), t[L][:, :3, 2]], 1) for L in JOINT_LINKS], 2)
        return pe, Re, J

    def solve(self, q, p_t, iters=60, lam=1e-3):
        eye = torch.eye(6, dtype=torch.float64, device=self.dev)
        for _ in range(iters):
            pe, Re, J = self.fk(q); e = torch.cat([p_t-pe, rotvec(self.R0@Re.transpose(1, 2))], 1)
            dq = (J.transpose(1, 2)@torch.linalg.solve(J@J.transpose(1, 2)+lam**2*eye, e[:, :, None]))[:, :, 0]
            q = q+dq*(0.3/dq.abs().amax(1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
        pe, Re, _ = self.fk(q)
        return q, (p_t-pe).norm(dim=1), rotvec(self.R0@Re.transpose(1, 2)).norm(dim=1)

    def project(self, q, p_t, iters=3, lam=1e-3):
        """Least-norm joint change that puts the gripper back on p_t (position only; orientation stays free)."""
        eye = torch.eye(3, dtype=torch.float64, device=self.dev)
        for _ in range(iters):
            pe, _, J = self.fk(q); Jp = J[:, :3]
            q = q+(Jp.transpose(1, 2)@torch.linalg.solve(Jp@Jp.transpose(1, 2)+lam**2*eye, (p_t-pe)[:, :, None]))[:, :, 0]
        return q


def relax(ik, label, q, p_t, pairs, q_min, q_max, iters=300, eta=0.05, w_smooth=0.5, clip=0.02):
    """Turn the poses of the IK grid towards clearance: gradient ascent on min(d, 5 cm) at every node minus w_smooth
    times the squared joint difference of 26-neighbours, each step followed by a position-only projection back onto the
    node's gripper position (the orientation becomes free) and the joint limits."""
    lo_t = torch.tensor(q_min, device=q.device); hi_t = torch.tensor(q_max, device=q.device); a, b = torch.tensor(pairs[:, 0], device=q.device), torch.tensor(pairs[:, 1], device=q.device)
    for _ in range(iters):
        qv = q.float().requires_grad_(True)
        obj = label.differentiable(qv).clamp(max=MARGIN).sum()-w_smooth*((qv[a]-qv[b])**2).sum()
        g = torch.nan_to_num(torch.autograd.grad(obj, qv)[0].double(), 0., 0., 0.)
        q = ik.project(torch.minimum(torch.maximum(q+(eta*g).clamp(-clip, clip), lo_t), hi_t), p_t)
    return q


def flood_fill(ik, Pt, gi, n, seed_node, seed_q, q_min, q_max):
    """IK at every grid point with ik's orientation, spread from the seed so neighbours stay on one branch."""
    dev = Pt.device; ngrid = len(gi); flat = lambda ijk: (ijk[..., 0]*n[1]+ijk[..., 1])*n[2]+ijk[..., 2]
    Q = np.zeros((ngrid, 6)); solved = np.zeros(ngrid, bool); tries = np.zeros(ngrid, int)
    q, pe, re = ik.solve(torch.tensor(seed_q[None], device=dev), Pt[[seed_node]], iters=300)
    if not (pe.item() < 1e-5 and re.item() < 1e-5 and np.all(np.abs(q.cpu().numpy()) <= q_max)): return Q, solved, 0
    Q[seed_node] = q[0].cpu().numpy(); solved[seed_node] = True; frontier = [seed_node]; waves = 0
    while frontier:
        cand = {}
        for i in frontier:
            for d in N26:
                j = gi[i]+d
                if np.all(j >= 0) and np.all(j < n):
                    jj = int(flat(j))
                    if not solved[jj] and tries[jj] < 4: cand.setdefault(jj, i)
        if not cand: break
        js = np.array(list(cand)); q0 = torch.tensor(Q[[cand[j] for j in js]], device=dev)
        q, pe, re = ik.solve(q0.clone(), Pt[js]); qn = q.cpu().numpy()
        ok = ((pe < 1e-4) & (re < 1e-3)).cpu().numpy() & np.all((qn >= q_min) & (qn <= q_max), 1) & (np.abs(qn-q0.cpu().numpy()).max(1) < JUMP)
        Q[js[ok]] = qn[ok]; solved[js[ok]] = True; tries[js[~ok]] += 1; frontier = list(js[ok]); waves += 1
    return Q, solved, waves


def load_field_module():
    spec = importlib.util.spec_from_file_location('ntrl_cabinet_field', ROOT/'experiments/ntrl_cabinet/field.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.NTRL, m.NORM = ROOT/'baselines/ntrl-demo', ROOT/'runtime/configs/joint_normalization.npz'; return m


def embed(net, x):
    """One side of NN.out: the instance-normalized 256-d embedding of release coordinates x (B, 6)."""
    xin = net.input_mapping(x)
    u = torch.sin(xin@net.lip_norm(net.pe_gate[0].weight).T+net.pe_gate[0].bias); v = torch.sin(xin@net.lip_norm(net.pe_gate[1].weight).T+net.pe_gate[1].bias)
    h = xin
    for ii in range(net.nl1):
        h_tmp = h
        for k in (1, 2):
            L = net.encoder[3*ii+k]; y = h@net.lip_norm(L.weight).T+L.bias; h = u*torch.sin(y)+v*(1-torch.sin(y))
        L = net.encoder[3*ii+3]; y = h@net.lip_norm(L.weight).T+L.bias; g = torch.sigmoid(0.1*net.gate[ii].weight); h = (1-g)*h_tmp+g*torch.sin(y)
    L = net.encoder[-1]; return net.encoder_norm(h@net.lip_norm(L.weight).T+L.bias)


def metric(e0, e1):
    d = torch.sqrt((e0-e1)**2+1e-6).view(len(e0), -1, 16)
    return 0.2*((torch.logsumexp(10*d, 2)-math.log(16))/10).sum(1)


def weights_blob(net):
    with torch.no_grad():
        parts = [(2*math.pi*net.B).float()]                                   # (6, 128)
        for L in list(net.pe_gate)+[net.encoder[k] for k in range(1, 3*net.nl1+1)]+[net.encoder[-1]]:
            parts += [net.lip_norm(L.weight).float(), L.bias.float()]
        parts.append(torch.stack([torch.sigmoid(0.1*net.gate[ii].weight).reshape(()) for ii in range(net.nl1)]).float())
    return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in parts]).astype(np.float32)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    dev = a.device; started = time.monotonic(); out = a.output; out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0); rng = np.random.default_rng(0)
    norm = np.load(R.NORMALIZATION); q_range = norm['q_range'].astype(np.float64); q_min, q_max = norm['q_min'].astype(np.float64), norm['q_max'].astype(np.float64)
    doc = yaml.safe_load(SCENE.read_text()); label = Label(dev); report = {}

    # 1. the pure-torch label against the cuRobo labels of the in-cabinet set
    X = np.load(CABINET_SET/'sampled_points.npy', mmap_mode='r'); Dc = np.load(CABINET_SET/'clearance.npy', mmap_mode='r'); pick = np.sort(rng.choice(len(X), 20000, replace=False))
    xs = torch.tensor(np.concatenate([X[pick, :6], X[pick, 6:]]), device=dev); ref = np.concatenate([Dc[pick, 0], Dc[pick, 1]])
    mine = label(xs*torch.tensor(q_range, dtype=torch.float32, device=dev)).cpu().numpy(); dev_label = np.abs(mine-ref)
    report['label_check'] = dict(configs=int(len(ref)), max_abs_m=float(dev_label.max()), p99_abs_m=float(np.quantile(dev_label, .99)))
    print('label check', report['label_check'], flush=True); assert dev_label.max() < 1e-4, 'pure-torch label does not reproduce the stored labels'

    # 2. grid and inverse kinematics, spread from the seed
    lo = np.array([doc['sampling'][k][0] for k in 'xyz']); hi = np.array([doc['sampling'][k][1] for k in 'xyz'])
    n = np.round((hi-lo)/H).astype(int)+1; ngrid = int(n.prod()); gi = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing='ij'), -1).reshape(-1, 3)
    P = lo+gi*H; flat = lambda ijk: (ijk[..., 0]*n[1]+ijk[..., 1])*n[2]+ijk[..., 2]
    suite = next(s for s in json.loads((TESTS/'suite_manifest.json').read_text())['suites'] if s['task'] == SEED[0])
    seed_q = np.load(TESTS/'suites'/suite['file'], allow_pickle=True)[SEED[1]][SEED[2]].astype(np.float64)
    t_seed = R.Kinematics().transforms(torch.tensor(seed_q[None]), [R.EE_LINK])[R.EE_LINK][0].numpy()
    gx = t_seed[:3, 0].copy(); gx[0] = 0; gx /= np.linalg.norm(gx); gz = np.array([1., 0., 0.]); R0 = np.stack([gx, np.cross(gz, gx), gz], 1)
    ik = IK(dev, R0); seed_node = int(flat(np.clip(np.round((t_seed[:3, 3]-lo)/H).astype(int), 0, n-1)))
    Pt = torch.tensor(P, dtype=torch.float64, device=dev); Q, solved, waves = flood_fill(ik, Pt, gi, n, seed_node, seed_q, q_min, q_max)
    assert solved.sum() > 1000, 'seed IK failed'
    ids = np.full(ngrid, -1, np.int32); nodes = np.nonzero(solved)[0]; ids[nodes] = np.arange(len(nodes)); N = len(nodes)
    report['grid'] = dict(shape=n.tolist(), points=ngrid, solved=int(N), waves=waves); print('ik', report['grid'], flush=True)
    Qn = torch.tensor(Q[nodes], device=dev); pairs = []
    for d in N26:
        if next(c for c in d if c) < 0: continue
        j = gi[nodes]+np.array(d); inb = np.all((j >= 0) & (j < n), 1); dst = np.full(N, -1); dst[inb] = ids[flat(j[inb])]; m = dst >= 0; pairs.append(np.stack([np.nonzero(m)[0], dst[m]], 1))
    pairs = np.concatenate(pairs); d_before = label(Qn.float()).cpu().numpy()
    Qn = relax(ik, label, Qn, Pt[nodes], pairs, q_min, q_max); Q[nodes] = Qn.cpu().numpy(); d_after = label(Qn.float()).cpu().numpy(); pe, Re, _ = ik.fk(Qn)
    report['relax'] = dict(free_before=float((d_before > 0).mean()), free_after=float((d_after > 0).mean()), clear5mm_before=float((d_before > .005).mean()), clear5mm_after=float((d_after > .005).mean()),
                           neighbour_jump_max_rad=float(np.abs(Q[nodes[pairs[:, 0]]]-Q[nodes[pairs[:, 1]]]).max()), approach_x_min=float(Re[:, 0, 2].min()), approach_x_median=float(Re[:, 0, 2].median()))
    report['ik_error'] = dict(position_max_m=float((pe-Pt[nodes]).norm(dim=1).max()))
    print('relax', report['relax'], report['ik_error'], flush=True)

    # 3. ground-truth speed at the nodes
    d_node = label(Qn.float()); s_node = target_speed(d_node)
    cup_box = np.array([[.56, .88], [.13, .53], [-.035, .665]]); inside = np.all((P[nodes] >= cup_box[:, 0]) & (P[nodes] <= cup_box[:, 1]), 1); dn = d_node.cpu().numpy()
    report['grid'].update(free_outside_cupboard=float((dn[~inside] > 0).mean()), free_inside_cupboard=float((dn[inside] > 0).mean()),
                          target_speed_median_outside=float(np.median(s_node.cpu().numpy()[~inside])), orientation=R0.round(4).tolist(), seed_q=seed_q.round(4).tolist())
    print('grid', report['grid'], flush=True)

    # 4. edges: 49 canonical directions of the 5x5x5 block, cost = int |dx| / s' over the straight joint-space segment
    edges = np.full((N, len(DIRS)), np.inf, np.float32); t = torch.linspace(0, 1, STEPS+1, dtype=torch.float64, device=dev); qr = torch.tensor(q_range, device=dev); dropped = 0
    for k, d in enumerate(DIRS):
        j = gi[nodes]+np.array(d); inb = np.all((j >= 0) & (j < n), 1); src = np.nonzero(inb)[0]; dst = ids[flat(j[inb])]; keep = dst >= 0; src, dst = src[keep], dst[keep]
        jump = np.abs(Q[nodes[dst]]-Q[nodes[src]]).max(1) > JUMP; dropped += int(jump.sum()); src, dst = src[~jump], dst[~jump]
        for s in range(0, len(src), 4000):
            a0, a1 = Qn[src[s:s+4000]], Qn[dst[s:s+4000]]
            seg = a0[:, None]+t[None, :, None]*(a1-a0)[:, None]                       # (b, 17, 6) radians
            inv = 1/target_speed(label(seg.reshape(-1, 6))).reshape(len(a0), -1)
            length = ((a1-a0)/qr).norm(dim=1); cost = length*torch.trapezoid(inv, t, dim=1)
            edges[src[s:s+4000], k] = cost.float().cpu().numpy()
    report['edges'] = dict(directions=len(DIRS), finite=int(np.isfinite(edges).sum()), dropped_jumps=dropped); print('edges', report['edges'], flush=True)

    # 5. reference Dijkstra and tightness against the direct segment, from the seed and from a point in front of the cupboard
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    src, kk = np.nonzero(np.isfinite(edges)); dst = ids[flat(gi[nodes[src]]+np.array(DIRS)[kk])]; w = edges[src, kk].astype(np.float64)
    G = coo_matrix((np.concatenate([w, w]), (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(N, N)).tocsr()
    presets = {'middle compartment': MIDDLE, 'top compartment': (0.705, 0.330, 0.530), 'bottom compartment': (0.705, 0.330, 0.090),
               'in front of the cupboard': (0.405, 0.330, 0.315), 'beside the cupboard': (0.405, 0.080, 0.165)}
    dn = d_node.cpu().numpy(); free_ids = np.nonzero(dn > OFFSET)[0]; preset_nodes = {}
    for name, xyz in presets.items():                                         # nearest grid pose with >= 5 mm clearance
        c = (np.array(xyz)-lo)/H; preset_nodes[name] = int(free_ids[np.argmin(np.abs(gi[nodes[free_ids]]-c).sum(1))])
    report['tightness'] = {}; cup_lo = np.array([.56, .13, -.035]); cup_hi = np.array([.88, .53, .665])
    in_cup = np.all((P[nodes] >= cup_lo) & (P[nodes] <= cup_hi), 1)
    for name in ('in front of the cupboard', 'middle compartment'):
        s0 = preset_nodes[name]; dist, pred = dijkstra(G, indices=s0, return_predecessors=True); ok = np.isfinite(dist); ok[s0] = False
        groups = {'outside, collision-free': ok & ~in_cup & (dn > OFFSET), 'cupboard, collision-free': ok & in_cup & (dn > OFFSET), 'in contact': ok & (dn <= 0)}
        report['tightness'][name] = {}
        for gname, m in groups.items():
            goals = rng.choice(np.nonzero(m)[0], min(96, int(m.sum())), replace=False); paths = []
            for g in goals:
                chain = [g]
                while chain[-1] != s0: chain.append(pred[chain[-1]])
                paths.append(Qn[chain[::-1]])
            _, after = optimize_paths(label, paths, qr); gain = 1-after.double().cpu().numpy()/dist[goals]
            report['tightness'][name][gname] = dict(paths=int(len(goals)), graph_time_median=float(np.median(dist[goals])), optimizer_gain_median=float(np.median(gain)),
                                                    optimizer_gain_p90=float(np.quantile(gain, .9)), optimizer_gain_max=float(gain.max()))
        if name == 'middle compartment': ref_dist = dist
    print('tightness', json.dumps(report['tightness'], indent=1), flush=True)

    # 6. learned fields: embeddings, effective weights, validation pairs
    FIELD = load_field_module(); xg = (Qn/qr).float(); checkpoints = {}
    val_pairs = np.stack([rng.integers(0, N, 48), rng.integers(0, N, 48)], 1); val_pairs[:8, 0] = preset_nodes['middle compartment']
    for key, (title, run) in CHECKPOINTS.items():
        net = FIELD.load_metric(ROOT/'results'/run/'training/checkpoint.pt', dev, 'metric', 'release').net
        with torch.no_grad(): E = torch.cat([embed(net, xg[s:s+4096]) for s in range(0, N, 4096)])
        x01 = torch.cat([xg[val_pairs[:, 0]], xg[val_pairs[:, 1]]], 1)
        with torch.enable_grad():
            tau, _, xp = net.out(x01); grad = torch.autograd.grad(tau.sum(), xp)[0]
        mine = metric(E[val_pairs[:, 0]], E[val_pairs[:, 1]]); consistency = float((mine-tau[:, 0]).abs().max())
        half = torch.tensor(E.cpu().numpy().astype(np.float16).astype(np.float32), device=dev)
        rnd = rng.integers(0, N, (20000, 2)); t32, t16 = metric(E[rnd[:, 0]], E[rnd[:, 1]]), metric(half[rnd[:, 0]], half[rnd[:, 1]])
        blob = weights_blob(net); (out/f'net_{key}.bin').write_bytes(blob.tobytes()); (out/f'emb_{key}.bin').write_bytes(E.cpu().numpy().astype(np.float16).tobytes())
        checkpoints[key] = dict(title=title, run=run, weights=int(len(blob)), embed_check_max_abs=consistency,
                                float16_T_rel_error_p99=float(np.quantile(((t16-t32).abs()/t32.clamp_min(1e-6)).cpu().numpy(), .99)),
                                validation=dict(T=tau[:, 0].tolist(), speed1=(1/grad[:, 6:].norm(dim=1)).tolist(), speed0=(1/grad[:, :6].norm(dim=1)).tolist(),
                                                e_head=E[val_pairs[0, 1], :8].tolist()))
        print('checkpoint', key, run, 'embed vs out', consistency, 'f16 T p99 rel', checkpoints[key]['float16_T_rel_error_p99'], flush=True)
        assert consistency < 1e-4

    # 7. files and meta
    nodes_blob = np.concatenate([gi[nodes].astype(np.float32), Q[nodes].astype(np.float32), d_node.cpu().numpy()[:, None].astype(np.float32),
                                 s_node.float().cpu().numpy()[:, None]], 1).astype(np.float32)                 # (N, 11): ix iy iz, q (6), d, s'
    (out/'nodes.bin').write_bytes(nodes_blob.tobytes()); (out/'edges.bin').write_bytes(edges.tobytes())
    root = ET_chain()
    meta = dict(grid=dict(lo=lo.tolist(), h=H, shape=n.tolist()), nodes=int(N), dirs=DIRS, orientation=R0.tolist(), seed=dict(suite=SEED[0], key=SEED[1], index=SEED[2], node=int(ids[seed_node])),
                presets=preset_nodes, q_range=q_range.tolist(), alpha=ALPHA, margin=MARGIN, offset=OFFSET, steps=STEPS, jump=JUMP,
                boxes=[dict(name=c['name'], center=c['pose'][:3], dims=c['dims']) for c in doc['cuboids']], sampling=doc['sampling'],
                compartments=doc['compartments'], chain=root, arm_links=ARM, checkpoints=checkpoints,
                validation=dict(pairs=val_pairs.tolist(), dijkstra_start=preset_nodes['middle compartment'],
                                dijkstra_sample=[[int(i), float(ref_dist[i]) if np.isfinite(ref_dist[i]) else -1.] for i in rng.choice(N, 400, replace=False)]),
                report=report, elapsed_seconds=time.monotonic()-started)
    (out/'meta.json').write_text(json.dumps(meta, separators=(',', ':'))); print('done', N, 'nodes', round(time.monotonic()-started), 's', flush=True)


def ET_chain():
    """Kinematic chain base_link -> grasp_frame for the page: per joint its origin (4x4, row-major), moving-joint index or -1, child link."""
    kin = R.Kinematics(); chain = []; link = R.EE_LINK
    while link != 'base_link':
        nidx, parent, idx = kin.joint_of[link]; chain.append(dict(child=link, moving=idx, origin=getattr(kin, f'origin_{nidx}').reshape(-1).tolist())); link = parent
    return chain[::-1]


if __name__ == '__main__': main()
