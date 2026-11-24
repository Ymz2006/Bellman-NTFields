"""Placements of the two arms in front of the measured cupboard: scene files and reach.

A placement is (dx, s, h): dx = distance from the bases' axis to the cupboard's front face (m), s = half spacing of the two bases
(left at y = +s, right at y = -s: symmetric about the cupboard's centre plane y = 0), h = base height above the floor. The scene is
jeffrey/real_world/make_scenes.py's (same cupboard, pedestals, sampling-box rule, zone boxes) with the pedestal tops lowered 0.04 m as
jeffrey/lbs_arm/make_real_scene.py does; (0.25, 0.30, 0.60) reproduces jeffrey/lbs_arm/scenes/real_world_12dof.yml.
Frame: origin midway between the bases, x towards the cupboard. Moving the arms back = a larger dx; the cupboard, the zone boxes and the
far end of the grasp-point sampling box move with the cupboard, the near end of the sampling box stays 0.08 m in front of the bases.

    python place.py scene --dx 0.55 --s 0.30 --h 0.60                 # writes scenes/<name>.yml, prints the path
    python place.py reach --out <dir> [--dx ...] [--s ...] [--h ...]   # reach of a grid of placements -> <dir>/reach.json

Reach = make_scenes.Reach (4 cm voxels of the zone interiors, flood fill, the other arm at home) per arm, and the share of
(left in zone i, right in zone j) pairs of reached configurations that are valid together.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = next(q for q in HERE.parents if (q / 'jeffrey/real_world/make_scenes.py').is_file())
sys.path.insert(0, str(ROOT / 'jeffrey/real_world'))
import make_scenes as MS  # noqa: E402
from robot import Arms, HOME  # noqa: E402

GAP = 0.04


def name_of(dx, s, h):
    return f'p_d{round(dx * 100):03d}_s{round(s * 100):02d}_h{round(h * 100):02d}'


def bases_of(s):
    return {'_left': (0.0, float(s), 0.0), '_right': (0.0, -float(s), 0.0)}


def scene_doc(dx, s, h):
    floor = -h
    Z = MS.zones(dx, floor)
    r4 = lambda v: [round(c, 4) for c in v]  # noqa: E731
    cub = MS.cuboids(dx, floor, bases_of(s))
    for c in cub:
        if c['name'].startswith('pedestal'):
            c['dims'][2] = round(c['dims'][2] - GAP, 6)
            c['pose'][2] = round(c['pose'][2] - GAP / 2, 6)
    return dict(robots=[dict(name='left', base=[0.0, float(s), 0.0, 1.0, 0.0, 0.0, 0.0]), dict(name='right', base=[0.0, -float(s), 0.0, 1.0, 0.0, 0.0, 0.0])],
                cuboids=cub, sampling=MS.sampling_box(dx, floor),
                compartments=[dict(name=k, x=[round(dx + 0.04, 4), round(v[0][1], 4)], y=r4(v[1]), z=r4(v[2])) for k, v in Z.items()],
                interiors=[dict(name=k, x=r4(v[0]), y=r4(v[1]), z=r4(v[2])) for k, v in Z.items()],
                placement=dict(dx=float(dx), half_spacing=float(s), base_height=float(h)))


def write_scene(dx, s, h, folder=HERE / 'scenes'):
    path = Path(folder) / (name_of(dx, s, h) + '.yml')
    head = (f'# {path.stem}: two UR5e + Robotiq 2F-140 in front of the measured cupboard; front face {dx} m from the bases\' axis, bases at y = +-{s} m,\n'
            f'# {h} m above the floor. Written by jeffrey/arm_placement/place.py; do not edit by hand.\n\n')
    path.write_text(head + yaml.safe_dump(scene_doc(dx, s, h), sort_keys=False, default_flow_style=None, width=200))
    return path


def reach(dx, s, h, arms, seeds, gen, rounds, dev):
    bases = bases_of(s)
    R = {arm: MS.Reach(arms, dict(bases=bases, arm=arm, dx=dx, floor=-h), dev) for arm in bases}
    for r in R.values():
        r.seed(seeds)
        r.grow(gen, rounds=rounds)
    hl, hr = R['_left'].hits(), R['_right'].hits()
    cl, cr, cu = MS.coverage(hl), MS.coverage(hr), MS.coverage(MS.union(hl, hr))
    both = {k: float((hl[k] & hr[k]).mean()) for k in hl}
    doc = scene_doc(dx, s, h)
    scene = Arms([r['base'] for r in doc['robots']], doc['cuboids'], dev)
    rng = np.random.default_rng(MS.SEED)
    pair = {}
    for zl, zr in itertools.product(MS.ZONES, repeat=2):
        L, Rr = R['_left'].configs(zl), R['_right'].configs(zr)
        n = min(len(L), len(Rr), 20000)
        if n == 0:
            pair[f'{zl}|{zr}'] = None
            continue
        q = torch.tensor(np.concatenate([L[rng.permutation(len(L))[:n]], Rr[rng.permutation(len(Rr))[:n]]], 1), device=dev)
        pair[f'{zl}|{zr}'] = float(scene.parts(q)['valid'].float().mean())
    return dict(name=name_of(dx, s, h), dx=dx, s=s, h=h, left=cl, right=cr, union=cu, both_arms=both, union_mean=sum(cu.values()) / 3, left_mean=sum(cl.values()) / 3,
                right_mean=sum(cr.values()) / 3, both_mean=sum(both.values()) / 3, pair_valid=pair, voxels=int(R['_left'].total))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['scene', 'reach'])
    p.add_argument('--dx', type=float, nargs='+', default=[0.25])
    p.add_argument('--s', type=float, nargs='+', default=[0.30])
    p.add_argument('--h', type=float, nargs='+', default=[0.60])
    p.add_argument('--out', type=Path)
    p.add_argument('--seeds', type=int, default=2_000_000)
    p.add_argument('--rounds', type=int, default=30)
    a = p.parse_args()
    if a.mode == 'scene':
        for dx, s, h in itertools.product(a.dx, a.s, a.h):
            print(write_scene(dx, s, h))
        return
    dev, t0 = 'cuda:0', time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    arms = Arms([(0, 0, 0, 1, 0, 0, 0)], [], device=dev)
    S = MS.Seeds(arms, a.seeds, dev)
    gen = torch.Generator(device=dev).manual_seed(MS.SEED + 1)
    rows = []
    for h, s, dx in itertools.product(a.h, a.s, a.dx):
        rows.append(reach(dx, s, h, arms, S, gen, a.rounds, dev))
        r = rows[-1]
        print(f'{r["name"]}: union {100 * r["union_mean"]:.1f} % (bottom/middle/top {[round(100 * r["union"][k]) for k in MS.ZONES]}), one arm {100 * r["left_mean"]:.1f} / {100 * r["right_mean"]:.1f} %, '
              f'both arms {100 * r["both_mean"]:.1f} %, {time.time() - t0:.0f}s', flush=True)
    (a.out / 'reach.json').write_text(json.dumps(dict(stage='arm-placement-reach', seeds=a.seeds, rounds=a.rounds, voxel=MS.VOXEL, placements=rows, seconds=time.time() - t0), indent=1))
    (a.out / 'log.txt').write_text(f'{len(rows)} placements in {time.time() - t0:.0f}s\n')


if __name__ == '__main__':
    main()
