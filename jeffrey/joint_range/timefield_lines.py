"""Line data for the time-field page (host, read-only): the clearance label at the 15 interior points of every
axis-aligned grid edge (each pose to its +x, +y and +z neighbour), on the same straight joint-space segment whose
16-step trapezoid gives that edge's ground-truth cost in timefield.py. The page uses them to draw travel time and speed
every 1.6 mm along the x, y and z lines through a picked grid pose.

Reads timefield.py's output folder and adds:
  lines.bin   float32 (N, 3, 15): clearance d (m) at t = k/16, k = 1..15, of the edge from node i along +axis
              (NaN where the neighbour has no pose)
  lines.json  checks: the trapezoid of these labels against the stored edge costs, how far the gripper strays from the
              straight grid line between two poses, and validation values for timefield_check.js (ground-truth
              line times from SciPy's Dijkstra, learned T and field speed from PyTorch at line points).

  CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/timefield_lines.py --data jeffrey/joint_range/timefield_viewer
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import timefield as TF
SUB = 16                                                   # samples per grid edge (the edge costs' trapezoid steps)


def speed_of(d, m):
    s = np.clip(d, m['offset'], m['margin'])/m['margin']
    return m['alpha']*s**2*(2-s)**2+1-m['alpha']


def line_points(m, gi_src, axis, G2N, LD):
    """The page's line through a grid pose along one axis (same order and breaks as buildLine in the page's core):
    per point (node a, node b, k) with k = 0 for the pose itself, and a run id that changes at every gap."""
    n = m['grid']['shape']; out = []; run = 0; flat = lambda c: (c[0]*n[1]+c[1])*n[2]+c[2]
    for idx in range(n[axis]):
        c = list(gi_src); c[axis] = idx; a = G2N[flat(c)]
        if a < 0: run += 1; continue
        out.append((a, -1, 0, run, idx))
        b = -1
        if idx+1 < n[axis]: c[axis] = idx+1; b = G2N[flat(c)]
        if b >= 0 and np.isfinite(LD[a, axis, 0]):
            out += [(a, b, k, run, idx) for k in range(1, SUB)]
        else: run += 1
    return out


def line_truth(m, pts, axis, dist, Q, D, LD, E, kdir, qr):
    """Ground-truth time at every line point: at a grid pose its Dijkstra time; between two grid poses the smaller of
    (a) the linear interpolation of their two times and (b) the time of reaching either pose and moving along the
    segment to the point (a real path; the segment's trapezoid rescaled to its stored edge cost). (b) alone overshoots
    in contact zones, where the slow segment is the only way in that it sees."""
    T = np.empty(len(pts)); S = np.empty(len(pts))
    for p, (a, b, k, _, _) in enumerate(pts):
        if b < 0: T[p] = dist[a]; S[p] = speed_of(D[a], m); continue
        dd = np.concatenate([[D[a]], LD[a, axis].astype(np.float64), [D[b]]]); inv = 1/speed_of(dd, m)
        L = np.linalg.norm((Q[b]-Q[a])/qr); part = np.concatenate([[0.], np.cumsum((inv[:-1]+inv[1:])/2/SUB)])*L
        scale = float(E[a, kdir])/part[-1]
        T[p] = min(dist[a]+part[k]*scale, dist[b]+(part[-1]-part[k])*scale, (1-k/SUB)*dist[a]+(k/SUB)*dist[b]); S[p] = speed_of(dd[k], m)
    return T, S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True); ap.add_argument('--device', default='cuda:0'); a = ap.parse_args()
    started = time.monotonic(); dev = a.device; rng = np.random.default_rng(1)
    m = json.loads((a.data/'meta.json').read_text()); N = m['nodes']; n = np.array(m['grid']['shape']); lo = np.array(m['grid']['lo']); h = m['grid']['h']
    A = np.fromfile(a.data/'nodes.bin', np.float32).reshape(N, 11); gi = np.rint(A[:, :3]).astype(int); Q = A[:, 3:9].astype(np.float64); D = A[:, 9].astype(np.float64)
    E = np.fromfile(a.data/'edges.bin', np.float32).reshape(N, len(m['dirs'])); qr = np.array(m['q_range'])
    flat = lambda ijk: (ijk[..., 0]*n[1]+ijk[..., 1])*n[2]+ijk[..., 2]; G2N = np.full(int(n.prod()), -1); G2N[flat(gi)] = np.arange(N)
    label = TF.Label(dev); ik = TF.IK(dev, m['orientation'])
    LD = np.full((N, 3, SUB-1), np.nan, np.float32); t = torch.linspace(0, 1, SUB+1, dtype=torch.float64, device=dev)
    cost_err, stray, report = [], [], {}
    for axis in range(3):
        d = np.zeros(3, int); d[axis] = 1; kdir = m['dirs'].index(d.tolist())
        j = gi+d; inb = np.all(j < n, 1); src = np.nonzero(inb)[0]; dst = G2N[flat(j[inb])]; keep = dst >= 0; src, dst = src[keep], dst[keep]
        for s in range(0, len(src), 4000):
            sa, sb = src[s:s+4000], dst[s:s+4000]; a0 = torch.tensor(Q[sa], device=dev); a1 = torch.tensor(Q[sb], device=dev)
            seg = a0[:, None]+t[None, :, None]*(a1-a0)[:, None]                                   # (b, 17, 6) radians
            dd = label(seg[:, 1:SUB].reshape(-1, 6)).reshape(len(sa), SUB-1).double()
            LD[sa, axis] = dd.float().cpu().numpy()
            full = torch.cat([torch.tensor(D[sa], device=dev)[:, None], dd, torch.tensor(D[sb], device=dev)[:, None]], 1)
            cost = ((a1-a0)/torch.tensor(qr, device=dev)).norm(dim=1)*torch.trapezoid(1/TF.target_speed(full), t, dim=1)
            ref = torch.tensor(E[sa, kdir], device=dev).double(); cost_err.append(((cost-ref).abs()/ref).cpu().numpy())
            pe = ik.fk(seg[:, 1:SUB].reshape(-1, 6))[0].reshape(len(sa), SUB-1, 3)
            p0 = torch.tensor(lo+gi[sa]*h, device=dev); nominal = p0[:, None]+t[None, 1:SUB, None]*torch.tensor(d*h, device=dev, dtype=torch.float64)
            off = pe-nominal; along = off[..., axis]; across = torch.sqrt((off**2).sum(-1)-along**2)
            stray.append(np.stack([across.amax(1).cpu().numpy(), along.abs().amax(1).cpu().numpy()], 1))
        report[f'edges_{"xyz"[axis]}'] = int(len(src))
    ce, st = np.concatenate(cost_err), np.concatenate(stray)
    report.update(cost_vs_edges_rel_max=float(ce.max()), cost_vs_edges_rel_p99=float(np.quantile(ce, .99)), gripper_off_line_max_m=float(st[:, 0].max()), gripper_off_line_p99_m=float(np.quantile(st[:, 0], .99)),
                  gripper_along_line_offset_max_m=float(st[:, 1].max()), gripper_along_line_offset_p99_m=float(np.quantile(st[:, 1], .99)))
    print('lines', report, flush=True); assert ce.max() < 1e-3, 'interior labels do not reproduce the edge costs'   # nodes.bin holds float32 joints: ~1e-4

    # validation for timefield_check.js: SciPy ground truth and PyTorch learned values on the lines through the middle preset
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    src, kk = np.nonzero(np.isfinite(E)); dst = G2N[flat(gi[src]+np.array(m['dirs'])[kk])]; w = E[src, kk].astype(np.float64)
    G = coo_matrix((np.concatenate([w, w]), (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(N, N)).tocsr()
    s0 = m['presets']['middle compartment']; dist = dijkstra(G, indices=s0)
    FIELD = TF.load_field_module(); val = dict(source=int(s0), lines=[], learned={})
    picks = []
    for axis in range(3):
        d = np.zeros(3, int); d[axis] = 1; kdir = m['dirs'].index(d.tolist()); pts = line_points(m, gi[s0], axis, G2N, LD)
        T, S = line_truth(m, pts, axis, dist, Q, D, LD, E, kdir, qr)
        val['lines'].append(dict(axis=axis, count=len(pts), T=[float(v) if np.isfinite(v) else -1. for v in T], S=S.tolist()))
        for p in rng.choice(len(pts), 12, replace=False):
            a_, b_, k, _, _ = pts[p]; x = Q[a_]/qr if b_ < 0 else (Q[a_]+(k/SUB)*(Q[b_]-Q[a_]))/qr; picks.append((axis, int(p), x))
    xs = torch.tensor(np.stack([np.concatenate([Q[s0]/qr, x]) for _, _, x in picks]), dtype=torch.float32, device=dev)
    for key, (title, run) in TF.CHECKPOINTS.items():
        net = FIELD.load_metric(TF.ROOT/'results'/run/'training/checkpoint.pt', dev, 'metric', 'release').net
        with torch.enable_grad():
            tau, _, xp = net.out(xs); grad = torch.autograd.grad(tau.sum(), xp)[0]
        val['learned'][key] = dict(points=[[ax, p] for ax, p, _ in picks], T=tau[:, 0].tolist(), S=(1/grad[:, 6:].norm(dim=1)).tolist())
    (a.data/'lines.bin').write_bytes(LD.tobytes())
    (a.data/'lines.json').write_text(json.dumps(dict(sub=SUB, report=report, validation=val, elapsed_seconds=time.monotonic()-started), separators=(',', ':')))
    print('done', round(time.monotonic()-started), 's', flush=True)


if __name__ == '__main__': main()
