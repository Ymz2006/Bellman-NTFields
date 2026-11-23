"""Build the real-world cupboard scenes from the measured cupboard, choose the robot base placement by reachability, and write
scenes/real_world_6dof.yml, scenes/real_world_12dof.yml, report/placement.json and report/samples.npz (viewer data).

Cupboard (measured, outer box including the panels): 1.82 m tall, 0.56 m deep, 0.92 m wide, standing on the floor, open towards the robot
(-x). Top and bottom panels 0.08 m, sides 0.03 m, back 0.03 m, two dividers (thickness not measured: DIVIDER_T, assumed 0.03 m like the
other thin panels) evenly spaced so the three zones have equal height.

Robots: the release UR5e + Robotiq 2F-140 on a PEDESTAL x PEDESTAL column from the floor to its base; base_link z = 0 is the pedestal top.
Scene frame: x towards the cupboard, z up, origin at the robot base (6 DOF) or midway between the two bases (12 DOF); the floor is at
z = -H, the cupboard's front face at x = DX, the cupboard centred on y = 0.

Reach of a placement (Reach): the zone interiors are cut into VOXEL cubes; a voxel is reached when some valid configuration of the arm
(robot.py: obstacle >= 0.005 m, self >= 0.002 m, other arm >= 0.005 m with the other arm at HOME) puts the grasp point in it, with any
orientation. Uniform joint samples in [-pi, pi]^6 seed one configuration per voxel; then a flood fill perturbs every stored configuration
(normal noise, 0.03 / 0.1 / 0.3 rad) and stores the valid ones that land in new voxels, until a round adds < 0.1 % of the voxels. Uniform
samples alone measure where random joint vectors land (deep voxels are rare), not what the arm can reach.
Coverage of a zone = reached voxels / voxels.

6 DOF: grid over H and DX, pick the largest mean zone coverage. 12 DOF: grid over H, DX and the half spacing S (bases at y = +-S),
coverage of the union of the two arms, pick the largest mean. Seed SEED.

    CUDA_DEVICE_ORDER=PCI_BUS_ID ~/.venvs/bellman-ntfield/bin/python make_scenes.py --device cuda:1
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from robot import Arms, HOME, OBS_VALID_M, SELF_VALID_M, INTER_VALID_M

HERE = Path(__file__).resolve().parent
# --- measured cupboard (metres) ----------------------------------------------------------------------------------------------------
HEIGHT, DEPTH, WIDTH = 1.82, 0.56, 0.92
TOP_T, BOTTOM_T, SIDE_T, BACK_T = 0.08, 0.08, 0.03, 0.03
DIVIDER_T = 0.03                       # not measured: assumed equal to the side and back panels
# --- robot mount ---------------------------------------------------------------------------------------------------------------
PEDESTAL = 0.25                        # square column under each base (UR5e base diameter 0.149 m)
VOXEL = 0.04
SEED = 20261121
METAL, BOARD, MOUNT, FLOOR = (150, 156, 166), (182, 188, 196), (90, 96, 104), (120, 120, 115)
ZONES = ('bottom', 'middle', 'top')
SIGMAS = (0.03, 0.1, 0.3)


def box(name, x, y, z, color=None):
    c = dict(name=name, pose=[round((x[0] + x[1]) / 2, 6), round((y[0] + y[1]) / 2, 6), round((z[0] + z[1]) / 2, 6), 1.0, 0.0, 0.0, 0.0],
             dims=[round(x[1] - x[0], 6), round(y[1] - y[0], 6), round(z[1] - z[0], 6)])
    if color:
        c['color'] = list(color)
    return c


def zones(dx, floor):
    """Interior boxes of the three zones (x, y, z ranges), bottom to top."""
    h = (HEIGHT - TOP_T - BOTTOM_T - 2 * DIVIDER_T) / 3
    x, y = (dx, dx + DEPTH - BACK_T), (-WIDTH / 2 + SIDE_T, WIDTH / 2 - SIDE_T)
    z0 = floor + BOTTOM_T
    return {name: (x, y, (z0 + i * (h + DIVIDER_T), z0 + i * (h + DIVIDER_T) + h)) for i, name in enumerate(ZONES)}


def cuboids(dx, floor, bases):
    """Floor, one pedestal per base, the cupboard panels (named shelf_* as in the cabinet scenes, so their scene_boxes() finds them)."""
    xf, xb = dx, dx + DEPTH
    yl, yr = -WIDTH / 2, WIDTH / 2
    zb, zt = floor, floor + HEIGHT
    Z = zones(dx, floor)
    out = [box('floor', (-1.5, 2.5), (-2.0, 2.0), (floor - 0.05, floor), FLOOR)]
    for name, (bx, by, bz) in bases.items():
        out.append(box(f'pedestal{name}', (bx - PEDESTAL / 2, bx + PEDESTAL / 2), (by - PEDESTAL / 2, by + PEDESTAL / 2), (floor, bz), MOUNT))
    out += [box('shelf_bottom', (xf, xb), (yl, yr), (zb, zb + BOTTOM_T), METAL),
            box('shelf_top', (xf, xb), (yl, yr), (zt - TOP_T, zt), METAL),
            box('shelf_back', (xb - BACK_T, xb), (yl, yr), (zb + BOTTOM_T, zt - TOP_T), METAL),
            box('shelf_side_right', (xf, xb - BACK_T), (yl, yl + SIDE_T), (zb + BOTTOM_T, zt - TOP_T), METAL),
            box('shelf_side_left', (xf, xb - BACK_T), (yr - SIDE_T, yr), (zb + BOTTOM_T, zt - TOP_T), METAL)]
    for i, upper in enumerate(ZONES[:2]):
        z = Z[upper][2][1]
        out.append(box(f'shelf_divider_{i + 1}', (xf, xb - BACK_T), (yl + SIDE_T, yr - SIDE_T), (z, z + DIVIDER_T), BOARD))
    return out


def wrap(q):
    """Pan and wrists to [-pi, pi]; the elbow (limits +-pi) is clamped."""
    q = torch.atan2(torch.sin(q), torch.cos(q))
    return q


class Seeds:
    """Uniform joint samples of one arm in its own base frame, self-valid, kept on the device and shared by every placement."""

    def __init__(self, arms, n, device, chunk=1 << 17):
        gen = torch.Generator(device=device).manual_seed(SEED)
        qs, es = [], []
        done = 0
        while done < n:
            m = min(chunk, n - done)
            q = (torch.rand(m, 6, generator=gen, device=device) * 2 - 1) * math.pi
            p, ee = arms.local_spheres(q)
            ok = arms.self_distance(p) >= SELF_VALID_M
            qs.append(q[ok]), es.append(ee[ok])
            done += m
        self.q, self.ee = torch.cat(qs), torch.cat(es)


class Reach:
    """Reached voxels of one arm at one placement: one stored configuration per voxel, flood fill (see the module docstring)."""

    def __init__(self, arms, P, device):
        self.arms, self.device = arms, device
        origin = np.array(P['bases'][P['arm']])
        Z = zones(P['dx'], P['floor'])
        arms.set_cuboids(cuboids(P['dx'], P['floor'], P['bases']))
        self.cc = arms.cc - torch.tensor(origin, device=device, dtype=torch.float32)
        self.ch = arms.ch.clone()
        home, _ = arms.local_spheres(torch.tensor([HOME], device=device))
        self.others = [home[0] + torch.tensor(np.subtract(b, origin), device=device, dtype=torch.float32)
                       for name, b in P['bases'].items() if name != P['arm']]
        self.lo, self.n, self.offset, self.Z = {}, {}, {}, Z
        total = 0
        for k, rng in Z.items():
            lo = np.array([r[0] for r in rng])
            n = np.ceil((np.array([r[1] for r in rng]) - lo) / VOXEL - 1e-9).astype(int)
            self.lo[k] = torch.tensor(lo - origin, device=device, dtype=torch.float32)
            self.n[k] = torch.tensor(n, device=device)
            self.offset[k] = (total, n)
            total += int(n.prod())
        self.total = total
        self.conf = torch.zeros(total, 6, device=device)
        self.filled = torch.zeros(total, dtype=torch.bool, device=device)
        self.history = []

    def voxel(self, ee):
        """Flat voxel index (-1 outside every zone)."""
        vid = torch.full((len(ee),), -1, dtype=torch.long, device=self.device)
        for k in self.lo:
            v = torch.floor((ee - self.lo[k]) / VOXEL).long()
            n = self.n[k]
            inz = ((v >= 0) & (v < n)).all(1)
            flat = self.offset[k][0] + (v[:, 0] * n[1] + v[:, 1]) * n[2] + v[:, 2]
            vid = torch.where(inz, flat, vid)
        return vid

    def valid(self, p):
        self.arms.cc, self.arms.ch = self.cc, self.ch
        good = self.arms.obstacle(p) >= OBS_VALID_M
        for o in self.others:
            good &= self.arms.inter(p, o[None].expand(len(p), -1, -1)) >= INTER_VALID_M
        return good

    def store(self, q, ee, chunk=1 << 16):
        vid = self.voxel(ee)
        idx = (vid >= 0).nonzero().squeeze(1)
        for i in range(0, len(idx), chunk):
            j = idx[i:i + chunk]
            ok = self.valid(self.arms.local_spheres(q[j])[0])
            v, qq = vid[j][ok], q[j][ok]
            if not len(v):
                continue
            # one configuration per voxel: index_put with repeated indices may mix the rows' joints on CUDA
            u, inv = torch.unique(v, return_inverse=True)
            first = torch.full((len(u),), len(v), device=self.device, dtype=torch.long).scatter_reduce(
                0, inv, torch.arange(len(v), device=self.device), 'amin')
            self.conf[u] = qq[first]
            self.filled[u] = True

    def seed(self, S):
        self.store(S.q, S.ee)
        self.history.append(int(self.filled.sum()))

    def grow(self, gen, copies=12, rounds=40, stop=0.001, chunk=1 << 17):
        for _ in range(rounds):
            before = int(self.filled.sum())
            base = self.conf[self.filled]
            q = base.repeat(copies, 1)
            sig = torch.tensor(SIGMAS, device=self.device).repeat_interleave(math.ceil(len(q) / len(SIGMAS)))[:len(q), None]
            q = q + sig * torch.randn(q.shape, generator=gen, device=self.device)
            q = wrap(q)
            for i in range(0, len(q), chunk):
                qq = q[i:i + chunk]
                p, ee = self.arms.local_spheres(qq)
                ok = self.arms.self_distance(p) >= SELF_VALID_M
                self.store(qq[ok], ee[ok])
            after = int(self.filled.sum())
            self.history.append(after)
            if after - before < stop * self.total:
                break
        return self

    def hits(self):
        out = {}
        f = self.filled.cpu().numpy()
        for k, (o, n) in self.offset.items():
            out[k] = f[o:o + int(n.prod())].reshape(tuple(n))
        return out

    def configs(self, k):
        o, n = self.offset[k]
        f = self.filled[o:o + int(n.prod())]
        return self.conf[o:o + int(n.prod())][f].cpu().numpy()


def coverage(hits):
    return {k: float(h.mean()) for k, h in hits.items()}


def union(a, b):
    return {k: a[k] | b[k] for k in a}


def sampling_box(dx, floor):
    """The cabinet scenes' sampling-box rule: height = cupboard, width = 1.5 x, depth = 2.5 x from the back face but x >= 0.08."""
    back = dx + DEPTH
    return dict(x=[round(max(0.08, back - 2.5 * DEPTH), 4), round(back, 4)], y=[round(-0.75 * WIDTH, 4), round(0.75 * WIDTH, 4)],
                z=[round(floor, 4), round(floor + HEIGHT, 4)])


def write_scene(path, header, robots, dx, H, bases):
    floor = -H
    Z = zones(dx, floor)
    r4 = lambda v: [round(c, 4) for c in v]  # noqa: E731
    doc = dict(robots=robots, cuboids=cuboids(dx, floor, bases), sampling=sampling_box(dx, floor),
               compartments=[dict(name=k, x=[round(dx + 0.04, 4), round(v[0][1], 4)], y=r4(v[1]), z=r4(v[2])) for k, v in Z.items()],
               interiors=[dict(name=k, x=r4(v[0]), y=r4(v[1]), z=r4(v[2])) for k, v in Z.items()])
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=140)
    path.write_text(''.join('# ' + line + '\n' if line else '#\n' for line in header.strip('\n').split('\n')) + '\n' + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--seeds', type=int, default=1_000_000, help='uniform joint samples that seed every placement of the search')
    ap.add_argument('--rounds', type=int, default=25, help='flood-fill rounds per placement of the search (stops early)')
    ap.add_argument('--final-seeds', type=int, default=4_000_000)
    ap.add_argument('--final-rounds', type=int, default=80)
    ap.add_argument('--quick', action='store_true', help='small grid, for testing')
    a = ap.parse_args()
    dev = a.device
    t0 = time.time()
    (HERE / 'scenes').mkdir(exist_ok=True)
    (HERE / 'report').mkdir(exist_ok=True)
    gen = torch.Generator(device=dev).manual_seed(SEED + 1)
    arms = Arms([(0, 0, 0, 1, 0, 0, 0)], [], device=dev)
    S = Seeds(arms, a.seeds, dev)
    print(f'seeds {len(S.q)} self-valid of {a.seeds}, {time.time() - t0:.0f}s', flush=True)
    Hs = [round(0.6 + 0.1 * i, 2) for i in range(6)]                       # base height above the floor
    DXs = [round(0.15 + 0.05 * i, 2) for i in range(13)]                   # base axis to the cupboard's front face
    Ss = [0.20, 0.25, 0.30, 0.35, 0.40, 0.46]                              # half spacing of the two bases (0.46 = cupboard's sides)
    if a.quick:
        Hs, DXs, Ss = Hs[2:5], DXs[1:7], Ss[1:3]

    def run(P, seeds=S, rounds=a.rounds):
        r = Reach(arms, P, dev)
        r.seed(seeds)
        return r.grow(gen, rounds=rounds)

    # ---- 6 DOF
    grid6 = []
    for H, dx in itertools.product(Hs, DXs):
        r = run(dict(bases={'': (0.0, 0.0, 0.0)}, arm='', dx=dx, floor=-H))
        c = coverage(r.hits())
        grid6.append(dict(H=H, dx=dx, **c, mean=sum(c.values()) / 3, rounds=len(r.history) - 1))
    best6 = max(grid6, key=lambda g: (round(g['mean'], 3), min(g[k] for k in ZONES)))
    print(f'6 DOF search {time.time() - t0:.0f}s best', best6, flush=True)

    # ---- 12 DOF: two arms, union of their reach (the other at HOME)
    grid12 = []
    DX12 = DXs[::2]                                                         # every 0.1 m for the 12-DOF grid
    for H, dx, s in itertools.product(Hs, DX12, Ss):
        bases = {'_left': (0.0, s, 0.0), '_right': (0.0, -s, 0.0)}
        hl, hr = (run(dict(bases=bases, arm=arm, dx=dx, floor=-H)).hits() for arm in bases)
        c = coverage(union(hl, hr))
        grid12.append(dict(H=H, dx=dx, S=s, **c, mean=sum(c.values()) / 3, left=sum(coverage(hl).values()) / 3, right=sum(coverage(hr).values()) / 3))
    best12 = max(grid12, key=lambda g: (round(g['mean'], 3), min(g[k] for k in ZONES)))
    print(f'12 DOF search {time.time() - t0:.0f}s best', best12, flush=True)

    # ---- the chosen placements again, more seeds and rounds (convergence check of the search values)
    del S
    torch.cuda.empty_cache()
    SF = Seeds(arms, a.final_seeds, dev)
    H6, dx6 = best6['H'], best6['dx']
    b6 = {'': (0.0, 0.0, 0.0)}
    F6 = run(dict(bases=b6, arm='', dx=dx6, floor=-H6), SF, a.final_rounds)
    H12, dx12, s12 = best12['H'], best12['dx'], best12['S']
    b12 = {'_left': (0.0, s12, 0.0), '_right': (0.0, -s12, 0.0)}
    F12 = [run(dict(bases=b12, arm=arm, dx=dx12, floor=-H12), SF, a.final_rounds) for arm in b12]
    c6, h12 = coverage(F6.hits()), union(F12[0].hits(), F12[1].hits())
    c12 = coverage(h12)
    print(f'final {time.time() - t0:.0f}s', c6, c12, flush=True)

    # ---- checks in the full scene model (robot.py, scene frame): every stored configuration is valid
    scene6 = Arms([(0, 0, 0, 1, 0, 0, 0)], cuboids(dx6, -H6, b6), device=dev)
    scene12 = Arms([(0, s12, 0, 1, 0, 0, 0), (0, -s12, 0, 1, 0, 0, 0)], cuboids(dx12, -H12, b12), device=dev)
    home = torch.tensor(HOME, device=dev)
    for k in ZONES:
        q6 = torch.tensor(F6.configs(k), device=dev)
        assert bool(scene6.parts(q6)['valid'].all()), k
        ql = torch.tensor(F12[0].configs(k), device=dev)
        assert bool(scene12.parts(torch.cat([ql, home.expand(len(ql), 6)], 1))['valid'].all()), k
        qr = torch.tensor(F12[1].configs(k), device=dev)
        assert bool(scene12.parts(torch.cat([home.expand(len(qr), 6), qr], 1))['valid'].all()), k
    home6, home12 = scene6.parts(home[None]), scene12.parts(torch.cat([home, home])[None])
    assert bool(home6['valid'][0]) and bool(home12['valid'][0])

    # 12-D examples with both arms in zones at once: a left configuration in zone i and a right one in zone j, valid together
    rng = np.random.default_rng(SEED)
    both, rate = {}, {}
    for zl, zr in itertools.product(ZONES, repeat=2):
        L, R = F12[0].configs(zl), F12[1].configs(zr)
        n = min(len(L), len(R), 20000)
        q = torch.tensor(np.concatenate([L[rng.permutation(len(L))[:n]], R[rng.permutation(len(R))[:n]]], 1), device=dev)
        ok = scene12.parts(q)['valid']
        both[f'{zl}|{zr}'] = q[ok][:16].cpu().numpy()
        rate[f'{zl}|{zr}'] = float(ok.float().mean())

    # ---- scene files
    Z6, Z12 = zones(dx6, -H6), zones(dx12, -H12)
    h = Z6['bottom'][2][1] - Z6['bottom'][2][0]
    pc = lambda c: ', '.join(f'{k} {100 * c[k]:.0f} %' for k in ZONES)  # noqa: E731
    cupboard = (f'Cupboard (measured): outer box {HEIGHT:.2f} m tall x {DEPTH:.2f} m deep x {WIDTH:.2f} m wide, panels included; top and bottom\n'
                f'{TOP_T:.2f} m metal, sides {SIDE_T:.2f} m, back {BACK_T:.2f} m, two dividers {DIVIDER_T:.2f} m (thickness ASSUMED, not measured)\n'
                f'evenly spaced: three zones {h:.4f} m tall, {DEPTH - BACK_T:.2f} m deep, {WIDTH - 2 * SIDE_T:.2f} m wide inside.\n'
                f'Open towards the robot (-x). Standing on the floor; each robot on a {PEDESTAL:.2f} x {PEDESTAL:.2f} m pedestal, base_link z = 0 = pedestal top.\n'
                'Arm model: the release UR5e + Robotiq 2F-140, cuRobo spheres (jeffrey/real_world/robot.py).')
    grid_text = (f'grid of base height {Hs[0]:.1f}-{Hs[-1]:.1f} m (0.1 m steps) and front-face distance {DXs[0]:.2f}-{DXs[-1]:.2f} m')
    write_scene(HERE / 'scenes/real_world_6dof.yml', f"""
real_world_6dof: one UR5e (6 joints) in front of the measured cupboard. Written by jeffrey/real_world/make_scenes.py; do not edit by hand.
{cupboard}
Frame: robot base_link at the origin, x towards the cupboard, z up. Floor z = {-H6:.2f} (base {H6:.2f} m above the floor); cupboard front face
x = {dx6:.2f}, back face x = {dx6 + DEPTH:.2f}, y {-WIDTH / 2:.2f} to {WIDTH / 2:.2f}, z {-H6:.2f} to {HEIGHT - H6:.2f}.
Placement: largest mean reached share of the zone interiors over a {grid_text} (make_scenes.py:
valid configurations, any gripper orientation, {VOXEL * 100:.0f} cm voxels). Reached: {pc(c6)}.
compartments: zone interiors from 0.04 m inside the opening (the cabinet scenes' convention); interiors: the full zone interiors.
sampling: the cabinet scenes' grasp-point box rule (height = cupboard, width 1.5 x, depth 2.5 x from the back face, x >= 0.08).
""", [dict(name='arm', base=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])], dx6, H6, b6)
    write_scene(HERE / 'scenes/real_world_12dof.yml', f"""
real_world_12dof: two UR5e arms (12 joints: left q[0:6], right q[6:12]) side by side in front of the measured cupboard, symmetric
about its width centre. Written by jeffrey/real_world/make_scenes.py; do not edit by hand.
{cupboard}
Frame: midway between the two bases, x towards the cupboard, z up. Bases at y = +-{s12:.2f} (left = +y), both facing +x.
Floor z = {-H12:.2f} (bases {H12:.2f} m above the floor); cupboard front face x = {dx12:.2f}, back face x = {dx12 + DEPTH:.2f}, y {-WIDTH / 2:.2f} to
{WIDTH / 2:.2f}, z {-H12:.2f} to {HEIGHT - H12:.2f}.
Placement: largest mean reached share of the zone interiors by either arm (the other at home, straight up) over a {grid_text}
and half spacing {Ss[0]:.2f}-{Ss[-1]:.2f} m. Reached by either arm: {pc(c12)}.
Validity: each arm against the cuboids (>= 0.005 m) and itself (>= 0.002 m); the two arms against each other (>= 0.005 m, all spheres).
""", [dict(name='left', base=[0.0, s12, 0.0, 1.0, 0.0, 0.0, 0.0]), dict(name='right', base=[0.0, -s12, 0.0, 1.0, 0.0, 0.0, 0.0])],
        dx12, H12, b12)

    # ---- viewer data and report
    def centres(hits, Z, reached=True):
        pts = []
        for k, hg in hits.items():
            lo = np.array([r[0] for r in Z[k]])
            pts.append(lo + (np.argwhere(hg if reached else ~hg) + 0.5) * VOXEL)
        return np.concatenate(pts).astype(np.float32)

    hl, hr = F12[0].hits(), F12[1].hits()
    save = {'6/reached': centres(F6.hits(), Z6), '6/missed': centres(F6.hits(), Z6, False),
            '12/reached': centres(h12, Z12), '12/missed': centres(h12, Z12, False),
            '12/reached_left': centres(hl, Z12), '12/reached_right': centres(hr, Z12)}
    for k in ZONES:
        q = F6.configs(k)
        save[f'6/examples/{k}'] = q[rng.permutation(len(q))[:24]]
    for k, q in both.items():
        save[f'12/examples/{k}'] = q
    np.savez_compressed(HERE / 'report/samples.npz', **save)
    search6 = next(g for g in grid6 if g['H'] == H6 and g['dx'] == dx6)
    search12 = next(g for g in grid12 if g['H'] == H12 and g['dx'] == dx12 and g['S'] == s12)
    rep = dict(seed=SEED, voxel=VOXEL, seeds=a.seeds, rounds=a.rounds, final_seeds=a.final_seeds, final_rounds=a.final_rounds,
               divider_assumed=DIVIDER_T, pedestal=PEDESTAL, grids=dict(H=Hs, dx=DXs, dx12=DX12, S=Ss),
               chosen6=dict(H=H6, dx=dx6, coverage=c6, search=search6, history=F6.history),
               chosen12=dict(H=H12, dx=dx12, S=s12, coverage=c12, left=coverage(hl), right=coverage(hr), search=search12,
                             history_left=F12[0].history, history_right=F12[1].history),
               both_arms_pair_valid_rate=rate,
               home_clearance=dict(six=dict(obstacle=float(home6['obstacle'].min()), self=float(home6['self'].min())),
                                   twelve=dict(obstacle=float(home12['obstacle'].min()), self=float(home12['self'].min()), inter=float(home12['inter'][0]))),
               grid6=grid6, grid12=grid12, seconds=round(time.time() - t0))
    (HERE / 'report/placement.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ('chosen6', 'chosen12', 'both_arms_pair_valid_rate', 'home_clearance', 'seconds')}, indent=1))


if __name__ == '__main__':
    main()
