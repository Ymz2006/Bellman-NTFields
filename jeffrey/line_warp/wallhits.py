"""What z-trained fields learned and where they cross walls, one process per maze.

* hits: re-plan planning seed 0 with the z planner on the fields of report.GRID and locate each arrived run's first contact (the stored rows hold no paths). report/mechanism.json['hits'][base].
* steps: the same frozen fields planned with a smaller MPPI step (temperature raised by the same factor, so the softmax is as selective per unit length): if a slow zone thin in steps is what lets
  the planner walk into walls, a smaller step should remove hits on the maps with thin zones and do little on the others. report/mechanism.json['steps'][base].
* field: the field's speed measured in z, on the same held-out poses for every field. The speed_fit of a field run is 1 / |dT/dx| over the quadrant stretch: right for a field trained
  in x, but for a field trained in z it scales with the map's local stretch even if the field is exact, so it cannot compare maps. report/mechanism.json['fieldz'][base].

    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/line_warp/wallhits.py hits|steps|field c22|m10|plus
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'jeffrey/maze_scale'), str(ROOT / 'jeffrey/cross_warp')]
import mechanism as M  # noqa: E402
import fit as FT  # noqa: E402

FIELD = {'m10k12t10': 'm10k12t10_zwB'}       # the rerun after the quadrant-stretch fix


def first_contact(maze, p, each=64):
    """Index of the first segment of path p (K, 3) that touches an obstacle, and the point of contact."""
    t = torch.linspace(0, 1, each, device=p.device)[None, :, None]
    x = p[:-1, None] + t * (p[1:, None] - p[:-1, None])
    d = maze.clearance(x.reshape(-1, 3)).reshape(len(p) - 1, each)
    bad = (d < 0).any(1).nonzero()
    if not len(bad):
        return None, None
    j = int(bad[0])
    return j, x[j, int((d[j] < 0).nonzero()[0])]


PLUS_FIELDS = dict(stripes=['stripeA_zw'], resnet=['resA4_zw'], tv=['bt10_zw'], recipe=['c22k1d_zw87358', 'c22k1d_zw1', 'c22k1d_zw2'], wiggly=['sep125_zw'], tight=['sep070_zw'], notv=['c22k1nt_zw87358'])


def fields_of(base, B):
    if base == 'plus':
        return M.PLUS, {k: [j for j in v if j in B] for k, v in PLUS_FIELDS.items()}
    prep, cells = M.WIDE[base]
    out = {}
    for name, job in cells.items():
        ids = [FIELD.get(job, f'{job}_zw87358'), f'{job}_zw1', f'{job}_zw2']
        out[name] = [j for j in ids if j in B]
    return prep, out


def field_z(base):
    """Per field: correlation of log 1 / |dT/dz| with the true log speed, the median speed in free space and at the floor, and how often the descent direction in z leads into the wall in the slow zone."""
    B = json.load(open(HERE / 'report/runs.json'))
    prep, cells = fields_of(base, B)
    maze, info = FT.load_maze(prep, 'cuda:0')
    import warp as W
    import plan as P
    import netwarp as NW
    import warpnet as WN
    import maze as MZ
    NW.install(W, P)
    hx = torch.tensor(np.load(prep / 'heldout/sampled_points.npy')[:30000], device='cuda:0')
    hd = torch.tensor(np.load(prep / 'heldout/clearance_cells.npy')[:30000, 0], device='cuda:0')
    S = maze.train_speed(hd)
    _, nrm = maze.clearance_normal(hx[:, :3])
    out = {}
    for name, ids in cells.items():
        rows = []
        for fid in ids:
            r = ROOT / 'results' / f'registered_{B[fid]}'
            net = P.load_field(r / 'training/checkpoint.pt')
            w = net.warp
            with torch.no_grad():
                z = torch.cat([w(hx[:, :3]), w(hx[:, 3:])], 1)
            net.bypass = True
            gs = []
            for i in range(0, len(z), 10000):
                with torch.enable_grad():
                    tau, _, zp = net.out(z[i:i + 10000])
                    g, = torch.autograd.grad(tau.sum(), zp)
                gs.append(g[:, :3].detach())
            net.bypass = False
            g = torch.cat(gs)
            sp = 1 / torch.linalg.vector_norm(g, dim=1).clamp(min=1e-6)
            lp, lt = torch.log(sp), torch.log(S)
            J = WN.jacobian_fd(w.plane, hx[:, :2].contiguous(), 0.02 / maze.E)
            nz = torch.linalg.solve(J.transpose(1, 2), nrm[:, :2, None])[:, :, 0]                   # d clearance / dz in the plane
            nz = torch.cat([nz, nrm[:, 2:3] / w.s_p], 1)
            into = (g * nz).sum(1) > 0                                                                # descent -g lowers the clearance
            slow = S < 0.3
            nn = torch.nn.functional.normalize(nrm[:, :2], dim=1)
            width = torch.einsum('nij,nj->ni', J, nn).norm(dim=1) * (MZ.MARGIN_CELLS / maze.E) / (0.015 * maze.planner_scale() * 6 * maze.E / 22.1)     # the slow zone in planner steps
            plane = nrm[:, :2].norm(dim=1) > 0.5
            by_width = {}
            for name_w, lo, hi in (('under 2', 0, 2), ('2 to 3', 2, 3), ('over 3', 3, 1e9)):
                sel = slow & plane & (width >= lo) & (width < hi)
                by_width[name_w] = dict(n=int(sel.sum()), into=float(into[sel].float().mean()) if bool(sel.any()) else None)
            rows.append(dict(job=fid, corr=float(torch.corrcoef(torch.stack([lp, lt]))[0, 1]), mae=float((lp - lt).abs().mean()), free=float(sp[S > 0.95].median()), wall=float(sp[S < 0.05].median()) if bool((S < 0.05).any()) else None,
                             into=float(into[slow].float().mean()), by_width=by_width))
            del net
            torch.cuda.empty_cache()
        if rows:
            out[name] = dict(seeds=rows, **{k: float(np.mean([x[k] for x in rows if x[k] is not None])) for k in ('corr', 'mae', 'free', 'wall', 'into')})
            pooled = {}
            for w in rows[0]['by_width']:
                n = sum(x['by_width'][w]['n'] for x in rows)
                pooled[w] = dict(n=n, into=sum(x['by_width'][w]['n'] * (x['by_width'][w]['into'] or 0) for x in rows) / n if n else None)
            out[name]['by_width'] = pooled
            print(name, {k: round(v, 3) for k, v in out[name].items() if k not in ('seeds', 'by_width')}, pooled, flush=True)
    path = HERE / 'report/mechanism.json'
    res = json.load(open(path))
    res.setdefault('fieldz', {})[base] = out
    json.dump(res, open(path, 'w'), indent=1)


def step_test(base, cells_used=('both', 'range', 'tv', 'dil'), factors=(1.0, 0.5, 0.25)):
    B = json.load(open(HERE / 'report/runs.json'))
    prep, cells = M.WIDE[base]
    maze, info = FT.load_maze(prep, 'cuda:0')
    import warp as W
    import plan as P
    import netwarp as NW
    import warpplan as WP
    import zplan as ZP
    NW.install(W, P)
    q = np.load(prep / 'queries.npz')
    S, G = torch.tensor(q['start'], device='cuda:0'), torch.tensor(q['goal'], device='cuda:0')
    tp = list(np.load(prep / 'truth_paths.npy', allow_pickle=True))
    st = P.settings(maze)
    out = {}
    for name in cells_used:
        job = cells[name]
        r = ROOT / 'results' / f'registered_{B[FIELD.get(job, f"{job}_zw87358")]}'
        net = P.load_field(r / 'training/checkpoint.pt')
        warp = ZP.field_warp(net)
        m = WP.quadrant_stretch(warp, maze)
        Lz = WP.path_length_z(warp, maze, tp)
        out[name] = {}
        for f in factors:
            s2 = dict(st, step=st['step'] * f, temperature=st['temperature'] / f)
            paths, stop, conv, used, pz = ZP.plan_z(net, maze, S, G, 0, Lz, m, s2)
            rows = P.score(maze, paths, conv)
            out[name][str(f)] = dict(solved=sum(x['success'] for x in rows), wall=sum(1 for x in rows if x['converged'] and x['collision']), not_arrived=sum(1 for x in rows if not x['converged']), runs=len(rows))
            print(name, f, out[name][str(f)], flush=True)
        del net
        torch.cuda.empty_cache()
    path = HERE / 'report/mechanism.json'
    res = json.load(open(path))
    res.setdefault('steps', {})[base] = out
    json.dump(res, open(path, 'w'), indent=1)


def main(base):
    prep, cells = M.WIDE[base]
    B = json.load(open(HERE / 'report/runs.json'))
    maze, info = FT.load_maze(prep, 'cuda:0')
    import warp as W
    import plan as P
    import netwarp as NW
    import warpnet as WN
    import warpplan as WP
    import zplan as ZP
    NW.install(W, P)
    q = np.load(prep / 'queries.npz')
    S, G = torch.tensor(q['start'], device='cuda:0'), torch.tensor(q['goal'], device='cuda:0')
    tp = list(np.load(prep / 'truth_paths.npy', allow_pickle=True))
    st = P.settings(maze)
    n, E, block = maze.n, maze.E, maze.describe().get('block', 8)
    inq = lambda c: (c < block) | (c >= n - block)
    out = {}
    for name, job in cells.items():
        fid = FIELD.get(job, f'{job}_zw87358')
        if fid not in B:
            continue
        r = ROOT / 'results' / f'registered_{B[fid]}'
        net = P.load_field(r / 'training/checkpoint.pt')
        warp = ZP.field_warp(net)
        m = WP.quadrant_stretch(warp, maze)
        Lz = WP.path_length_z(warp, maze, tp)
        paths, stop, conv, used, pz = ZP.plan_z(net, maze, S, G, 0, Lz, m, st)
        rows = P.score(maze, paths, conv)
        hit = dict(last_hop=0, jump=0, walked_in=0)
        step_cells, quad, local = [], [], []
        for i, row in enumerate(rows):
            if not (row['converged'] and row['collision']):
                continue
            p = paths[i]
            j, c = first_contact(maze, p)
            if j is None:
                continue
            ends = maze.clearance(p[j:j + 2])
            kind = 'last_hop' if j == len(p) - 2 else 'jump' if bool((ends >= 0).all()) else 'walked_in'
            hit[kind] += 1
            step_cells.append(float(torch.linalg.vector_norm(p[j + 1, :2] - p[j, :2]) / maze.pitch))
            Jc = WN.jacobian_fd(warp.plane, c[None, :2].contiguous(), 0.02 / E)[0]
            local.append(float((Jc[0, 0] * Jc[1, 1] - Jc[0, 1] * Jc[1, 0]).abs().sqrt()) / m)
            cx, cy = float(c[0]) * E + n / 2, float(c[1]) * E + n / 2
            quad.append(bool(inq(cx) and inq(cy)))
        allstep = torch.cat([torch.linalg.vector_norm(p[1:-1, :2] - p[:-2, :2], dim=1) for p in paths if len(p) > 3]) / maze.pitch
        out[name] = dict(job=fid, solved=sum(x['success'] for x in rows), runs=len(rows), arrived_wall=sum(1 for x in rows if x['converged'] and x['collision']), not_arrived=sum(1 for x in rows if not x['converged']), **hit,
                         stretch_at_contact_median=float(np.median(local)) if local else None, step_at_contact_median=float(np.median(step_cells)) if step_cells else None, step_all_median=float(allstep.median()), step_all_q95=float(allstep.quantile(0.95)), contact_in_quadrant=float(np.mean(quad)) if quad else None)
        print(name, out[name], flush=True)
        del net
        torch.cuda.empty_cache()
    path = HERE / 'report/mechanism.json'
    res = json.load(open(path))
    res.setdefault('hits', {})[base] = out
    json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    dict(hits=main, steps=step_test, field=field_z)[sys.argv[1]](sys.argv[2])
