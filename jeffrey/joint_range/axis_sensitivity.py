"""Axis sensitivity of the true travel-time field in joint space (host, read-only analysis on the Test 4 viewer grid).

Question (2026-11-01): the mixed-pairs model sees the back panel (T rises along +x) but not the side panels or shelf
boards (T flat along y, z). Are y and z intrinsically harder in joint space? Measure, at every grid pose p in the
cupboard, how much the TRUE time field T(q) must change when the gripper moves one metre in +x, +y or +z:

  joint velocity   dq_d = IK(e_d)                 three variants: least-norm position IK J_p^+ e_d, orientation-fixed
                                                  J^-1 [e_d; 0], and the grid's own poses (Q[p+d] - Q[p-d]) / 2h
  joint gradient   dT/dq                          eikonal identity of the true field: |dT/dx| = 1/s'(x) along the arrival
                                                  direction u of the optimal path (Dijkstra predecessor), x = q / q_range
  directional rate g_d = dT/dq . dq_d             time per metre of gripper motion along d; vector (g_x, g_y, g_z), |g|
  per unit joint motion g_d / |dx_d|              time per unit of network-input motion = cos(u, dx_d) / s'
  stretch          |dx_d| = |dq_d / q_range|      how far the pose moves in the network's input space per metre along d

Checks: dT/dq . dq_grid against the central difference of the Dijkstra times along the grid line (the slope the viewer
draws); learned-field rates from the checkpoints' autograd gradient with the same dq_d.

  CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/axis_sensitivity.py --data jeffrey/joint_range/test4_viewer/cabinet --output jeffrey/joint_range/axis_sensitivity
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import timefield as TF
from test4_field import SCENES

AX = 'xyz'


def q50(v): return float(np.median(v)) if len(v) else float('nan')
def iqr(v): return [float(np.quantile(v, .25)), float(np.quantile(v, .75))] if len(v) else [float('nan')]*2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--models', nargs='*', default=[], help='extra checkpoints as key=registered_<run> (trained in the same scene), added to the scene\'s Test 4 models'); a = ap.parse_args()
    t0 = time.monotonic(); dev = a.device; out = a.output; out.mkdir(parents=True, exist_ok=True)
    m = json.loads((a.data/'meta.json').read_text()); N = m['nodes']; n = np.array(m['grid']['shape']); lo = np.array(m['grid']['lo']); h = m['grid']['h']; qr = np.array(m['q_range'])
    A = np.fromfile(a.data/'nodes.bin', np.float32).reshape(N, 11); gi = np.rint(A[:, :3]).astype(int); Q = A[:, 3:9].astype(np.float64); D = A[:, 9].astype(np.float64); Sp = A[:, 10].astype(np.float64)
    E = np.fromfile(a.data/'edges.bin', np.float32).reshape(N, len(m['dirs'])); P = lo+gi*h
    flat = lambda ijk: (ijk[..., 0]*n[1]+ijk[..., 1])*n[2]+ijk[..., 2]; G2N = np.full(int(n.prod()), -1); G2N[flat(gi)] = np.arange(N)
    src, kk = np.nonzero(np.isfinite(E)); dst = G2N[flat(gi[src]+np.array(m['dirs'])[kk])]; w = E[src, kk].astype(np.float64)
    G = coo_matrix((np.concatenate([w, w]), (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(N, N)).tocsr()
    X = Q/qr                                                                    # network input coordinates

    # regions
    cup_lo, cup_hi = np.array(m['cupboard'][0]), np.array(m['cupboard'][1]); in_cup = np.all((P >= cup_lo-1e-9) & (P <= cup_hi+1e-9), 1)
    comp = np.full(N, -1)
    for ci, c in enumerate(m['compartments']):
        box = np.array([c['x'], c['y'], c['z']]); comp[np.all((P >= box[:, 0]-1e-9) & (P <= box[:, 1]+1e-9), 1)] = ci
    clear_class = np.where(D > m['offset'], 'free', np.where(D > 0, 'near', 'contact'))

    # 1. kinematics: geometric Jacobian at every pose, three joint-velocity fields per axis (rad per metre)
    ik = TF.IK(dev, m['orientation']); Qt = torch.tensor(Q, device=dev); pe, Re, J = ik.fk(Qt)
    assert float((pe.cpu().numpy()-P).__abs__().max()) < 2e-3, 'poses do not sit on the grid'
    lam = 1e-3; eye3 = torch.eye(3, dtype=torch.float64, device=dev); eye6 = torch.eye(6, dtype=torch.float64, device=dev)
    Jp = J[:, :3]; M_ln = Jp.transpose(1, 2)@torch.linalg.solve(Jp@Jp.transpose(1, 2)+lam**2*eye3, eye3.expand(N, 3, 3))      # (N, 6, 3): columns dq_x, dq_y, dq_z
    M_fix = (J.transpose(1, 2)@torch.linalg.solve(J@J.transpose(1, 2)+lam**2*eye6, eye6.expand(N, 6, 6)))[:, :, :3]            # orientation held
    sv = torch.linalg.svdvals(J); cond = (sv[:, 0]/sv[:, -1]).cpu().numpy()
    dq = {'ik_leastnorm': M_ln.cpu().numpy(), 'ik_fixed_orientation': M_fix.cpu().numpy()}
    grid_dq = np.full((N, 6, 3), np.nan); grid_ok = np.zeros((N, 3), bool)
    for ax in range(3):
        d = np.zeros(3, int); d[ax] = 1; jp, jm = gi+d, gi-d
        okp = np.all(jp < n, 1); okm = np.all(jm >= 0, 1); ip = np.where(okp, G2N[flat(np.clip(jp, 0, n-1))], -1); im = np.where(okm, G2N[flat(np.clip(jm, 0, n-1))], -1)
        ok = (ip >= 0) & (im >= 0); ok &= np.abs(Q[np.clip(ip, 0, N-1)]-Q[np.clip(im, 0, N-1)]).max(1) < 2*m['jump']
        grid_dq[ok, :, ax] = (Q[ip[ok]]-Q[im[ok]])/(2*h); grid_ok[:, ax] = ok
    dq['grid_central'] = grid_dq
    stretch = {k: np.linalg.norm(v/qr[None, :, None], axis=1) for k, v in dq.items()}                                             # |dx_d| per metre, (N, 3)
    rad_per_m = {k: np.linalg.norm(v, axis=1) for k, v in dq.items()}

    # 2. learned fields (loaded once): gradients at every pose for each start
    runs = {key: run for key, (_, run, _) in SCENES[m['scene']]['models'].items()}; runs.update(kv.split('=', 1) for kv in a.models)
    FIELD = TF.load_field_module(); nets = {key: FIELD.load_metric(TF.ROOT/'results'/run/'training/checkpoint.pt', dev, 'metric', 'release').net for key, run in runs.items()}
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)

    def learned(net, s0):
        T = np.empty(N); Gx = np.empty((N, 6))
        for s in range(0, N, 4096):
            x01 = torch.cat([Xt[[s0]].expand(len(Xt[s:s+4096]), 6), Xt[s:s+4096]], 1)
            with torch.enable_grad():
                tau, _, xp = net.out(x01); g = torch.autograd.grad(tau.sum(), xp)[0]
            T[s:s+4096] = tau[:, 0].detach().cpu().numpy(); Gx[s:s+4096] = g[:, 6:].cpu().numpy()
        return T, Gx/qr                                                                                                    # dT/dq (per rad)

    results = dict(data=str(a.data), scene=m['scene'], nodes=N, in_cupboard=int(in_cup.sum()), models=runs,
                   jacobian_condition=dict(median=q50(cond), p90=float(np.quantile(cond, .9)), max=float(cond.max())), starts={}); rows = []
    for sname in ('middle compartment', 'in front of the cupboard'):
        s0 = m['presets'][sname]; dist, pred = dijkstra(G, indices=s0, return_predecessors=True); reach = np.isfinite(dist) & (pred >= 0)
        # eikonal gradient of the true field: u / s' along the arrival direction, in x; then per radian
        u = np.zeros((N, 6)); pr = np.clip(pred, 0, N-1); du = X-X[pr]; nu = np.linalg.norm(du, axis=1); u[reach] = du[reach]/nu[reach, None]
        gradT_x = u/Sp[:, None]; gradT_q = gradT_x/qr
        rate = {k: np.einsum('ni,nid->nd', gradT_q, v) for k, v in dq.items()}                                           # g_d, time per metre, (N, 3)
        cosine = {k: np.einsum('ni,nid->nd', u, v/qr[None, :, None])/np.maximum(stretch[k], 1e-12) for k, v in dq.items()}
        # check: central difference of the Dijkstra time along the grid line
        fd = np.full((N, 3), np.nan)
        for ax in range(3):
            d = np.zeros(3, int); d[ax] = 1; ok = grid_ok[:, ax]; ip = G2N[flat(gi[ok]+d)]; im = G2N[flat(gi[ok]-d)]
            v = (dist[ip]-dist[im])/(2*h); v[~(np.isfinite(dist[ip]) & np.isfinite(dist[im]))] = np.nan; fd[ok, ax] = v
        L = {key: learned(net, s0) for key, net in nets.items()}
        lrate = {key: {k: np.einsum('ni,nid->nd', Gq, v) for k, v in dq.items()} for key, (_, Gq) in L.items()}
        lgrad_x = {key: np.linalg.norm(Gq*qr, axis=1) for key, (_, Gq) in L.items()}
        lcos = {key: {k: np.einsum('ni,nid->nd', Gq*qr, v/qr[None, :, None])/np.maximum(stretch[k]*lgrad_x[key][:, None], 1e-12) for k, v in dq.items()} for key, (_, Gq) in L.items()}
        lfd = {key: np.full((N, 3), np.nan) for key in nets}
        for ax in range(3):
            d = np.zeros(3, int); d[ax] = 1; ok = grid_ok[:, ax]; ip = G2N[flat(gi[ok]+d)]; im = G2N[flat(gi[ok]-d)]
            for key in nets: lfd[key][ok, ax] = (L[key][0][ip]-L[key][0][im])/(2*h)

        def summary(mask):
            mask = mask & reach; o = dict(poses=int(mask.sum()), true_T_median=q50(dist[mask]), true_grad_x_median=q50(1/Sp[mask]), learned_grad_x_median={key: q50(lgrad_x[key][mask]) for key in nets})
            for k in dq:
                mk = {ax: mask & np.isfinite(rate[k][:, ax]) for ax in range(3)}
                o[k] = {AX[ax]: dict(rad_per_m=q50(rad_per_m[k][mk[ax], ax]), stretch_x_per_m=q50(stretch[k][mk[ax], ax]),
                                    true_rate_abs=q50(np.abs(rate[k][mk[ax], ax])), true_rate_abs_iqr=iqr(np.abs(rate[k][mk[ax], ax])), true_rate_signed=q50(rate[k][mk[ax], ax]),
                                    true_per_unit_joint=q50(np.abs(rate[k][mk[ax], ax])/np.maximum(stretch[k][mk[ax], ax], 1e-12)), cos_abs=q50(np.abs(cosine[k][mk[ax], ax])),
                                    learned_rate_abs={key: q50(np.abs(lrate[key][k][mk[ax], ax])) for key in nets}, learned_cos_abs={key: q50(np.abs(lcos[key][k][mk[ax], ax])) for key in nets},
                                    learned_over_true={key: q50(np.abs(lrate[key][k][mk[ax], ax])/np.maximum(np.abs(rate[k][mk[ax], ax]), 1e-9)) for key in nets}) for ax in range(3)}
                allok = mask & np.all(np.isfinite(rate[k]), 1)
                o[k]['vector_norm'] = dict(true=q50(np.linalg.norm(rate[k][allok], axis=1)), learned={key: q50(np.linalg.norm(lrate[key][k][allok], axis=1)) for key in nets})
                if k == 'grid_central':
                    fo = allok & np.all(np.isfinite(fd), 1)
                    o[k]['check_vs_central_difference'] = {AX[ax]: dict(fd_abs_median=q50(np.abs(fd[fo, ax])), fd_signed_median=q50(fd[fo, ax]), ratio_median=q50(np.abs(rate[k][fo, ax])/np.maximum(np.abs(fd[fo, ax]), 1e-9)),
                                                                     spearman=float(np.corrcoef(np.argsort(np.argsort(np.abs(fd[fo, ax]))), np.argsort(np.argsort(np.abs(rate[k][fo, ax]))))[0, 1]),
                                                                     fd_per_unit_joint=q50(np.abs(fd[fo, ax])/np.maximum(stretch[k][fo, ax], 1e-12)),
                                                                     learned_fd_abs={key: q50(np.abs(lfd[key][fo, ax])) for key in nets}, learned_fd_signed={key: q50(lfd[key][fo, ax]) for key in nets},
                                                                     learned_fd_over_true_fd={key: q50(np.abs(lfd[key][fo, ax])/np.maximum(np.abs(fd[fo, ax]), 1e-9)) for key in nets}) for ax in range(3)}
            return o
        S = dict(start_node=int(s0), start_position=P[s0].round(4).tolist(), reachable=int(reach.sum()), regions={})
        S['regions']['cupboard, all'] = summary(in_cup)
        for cls in ('free', 'near', 'contact'): S['regions'][f'cupboard, {cls}'] = summary(in_cup & (clear_class == cls))
        for ci, c in enumerate(m['compartments']): S['regions'][f'{c["name"]} compartment, free'] = summary((comp == ci) & (clear_class == 'free'))
        for ci, c in enumerate(m['compartments']): S['regions'][f'{c["name"]} compartment, contact'] = summary((comp == ci) & (clear_class == 'contact'))
        S['regions']['outside cupboard, free'] = summary(~in_cup & (clear_class == 'free'))
        S['regions']['outside cupboard, contact'] = summary(~in_cup & (clear_class == 'contact'))
        # per-pose rows on the three grid lines through the start (what the viewer draws), least-norm IK
        if sname == 'middle compartment':
            for ax in range(3):
                for idx in range(n[ax]):
                    c = gi[s0].copy(); c[ax] = idx; i = G2N[flat(c)]
                    if i < 0 or not reach[i] and i != s0: continue
                    rows.append(dict(line=AX[ax], node=int(i), position=P[i].round(4).tolist(), clearance=round(float(D[i]), 4), true_T=round(float(dist[i]), 4),
                                     **{f'learned_T_{key}': round(float(L[key][0][i]), 4) for key in nets},
                                     **{f'rad_per_m_{AX[d]}': round(float(rad_per_m['ik_leastnorm'][i, d]), 3) for d in range(3)},
                                     **{f'stretch_{AX[d]}': round(float(stretch['ik_leastnorm'][i, d]), 4) for d in range(3)},
                                     **{f'true_rate_{AX[d]}': round(float(rate['ik_leastnorm'][i, d]), 3) for d in range(3)},
                                     **{f'fd_rate_{AX[d]}': (round(float(fd[i, d]), 3) if np.isfinite(fd[i, d]) else None) for d in range(3)},
                                     **{f'learned_rate_{key}_{AX[d]}': round(float(lrate[key]['ik_leastnorm'][i, d]), 3) for key in nets for d in range(3)},
                                     **{f'learned_fd_{key}_{AX[d]}': (round(float(lfd[key][i, d]), 3) if np.isfinite(lfd[key][i, d]) else None) for key in nets for d in range(3)}))
        results['starts'][sname] = S
        print(sname, 'done', round(time.monotonic()-t0), 's', flush=True)
    results['elapsed_seconds'] = time.monotonic()-t0
    (out/'results.json').write_text(json.dumps(results, indent=1)); (out/'middle_lines.json').write_text(json.dumps(rows, indent=0))
    # per-pose arrays for later plotting
    np.savez_compressed(out/'per_pose.npz', position=P, clearance=D, target_speed=Sp, in_cupboard=in_cup, compartment=comp, jac_cond=cond,
                        **{f'dq_{k}': v for k, v in dq.items()}, **{f'stretch_{k}': v for k, v in stretch.items()})
    print('done', round(time.monotonic()-t0), 's', flush=True)


if __name__ == '__main__': main()
