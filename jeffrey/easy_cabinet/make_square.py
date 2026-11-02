"""Derive the square scene, experiments/cabinet/scenes/cabinet_square.yml: three copies of the easy cupboard (cabinet_easy:
2 x wide compartments, 0.10 m panels) around the robot, each centred on its own axis and open towards the robot: +x (as in
cabinet_easy, lateral centre moved to y 0), +y (rotated +90 deg about z: opening faces -y) and -y (rotated -90 deg).
Axis-aligned boxes stay axis-aligned under 90 deg turns, so the rotated copies are the same cuboids with x and y
extents swapped and positions mapped (x, y) -> (-y, x) or (y, -x). Three table slabs (one under every cupboard, none
under the robot) give the floor. Sampling box: x and y in [-0.96, 0.96], z from the table top to the cupboard top.
`focus:` (the region that takes the cupboard's place in the coverage strata and the test regions) is the bounding box
of the three cupboards, so the `free` test region is the strip behind the robot (x < -0.46) and the corner gaps.
Compartments: the nine interiors, named <name>_px / _py / _ny; compartment strata draw one of the nine uniformly, so
most compartment-to-compartment queries cross between cupboards (the long-distance test).

    python3 jeffrey/easy_cabinet/make_square.py           # writes the scene
    python3 jeffrey/easy_cabinet/make_square.py --check   # derives it again and compares with the file
"""
import argparse, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]; SCENES = ROOT/'experiments/cabinet/scenes'
sys.path.insert(0, str(Path(__file__).resolve().parent)); import make_easy as ME
NAME = 'cabinet_square'; R_ = ME.STRONG.r


def turn(c, how):
    """Cuboid c mapped by a 90 deg turn about z: 'py' (x, y) -> (-y, x), 'ny' (x, y) -> (y, -x); dims x <-> y."""
    x, y, z = c['pose'][:3]; dx, dy, dz = c['dims']
    nx, ny = (-y, x) if how == 'py' else (y, -x)
    return dict(name=c['name']+'_'+how, pose=[R_(nx), R_(ny), R_(z)]+c['pose'][3:], dims=[R_(dy), R_(dx), R_(dz)], **({'color': c['color']} if 'color' in c else {}))


def turn_box(b, how):
    (x0, x1), (y0, y1) = b['x'], b['y']
    nx, ny = ((-y1, -y0), (x0, x1)) if how == 'py' else ((y0, y1), (-x1, -x0))
    return dict(name=b['name']+'_'+how, x=[R_(nx[0]), R_(nx[1])], y=[R_(ny[0]), R_(ny[1])], z=b['z'])


def derive():
    base, wide_doc, d = ME.derive(); yc = d['yc']
    home = []
    for c in d['cuboids']:
        if c['name'] == 'table': continue
        c = dict(c); c['pose'] = [c['pose'][0], R_(c['pose'][1]-yc), c['pose'][2]]+c['pose'][3:]; home.append(dict(c, name=c['name']+'_px'))
    comps = [dict(b, y=[R_(b['y'][0]-yc), R_(b['y'][1]-yc)], name=b['name']+'_px') for b in d['compartments']]
    cub = list(home)
    for how in ('py', 'ny'):
        cub += [turn(dict(c, name=c['name'][:-3]), how) for c in home]; comps += [turn_box(b, how) for b in d['compartments']]
    table = next(c for c in d['cuboids'] if c['name'] == 'table'); tz, th = table['pose'][2], table['dims'][2]
    lo = [min(c['pose'][i]-c['dims'][i]/2 for c in home) for i in range(3)]; hi = [max(c['pose'][i]+c['dims'][i]/2 for c in home) for i in range(3)]
    slab = dict(name='table', pose=[R_((lo[0]+hi[0])/2+0.0), 0.0, tz, 1.0, 0.0, 0.0, 0.0], dims=[R_(hi[0]-lo[0]+0.2), R_(hi[1]-lo[1]+0.2), th])
    tables = [slab, turn(dict(slab, name='table'), 'py'), turn(dict(slab, name='table'), 'ny')]
    cuboids = tables+cub; floor = tz+th/2
    allo = [min(c['pose'][i]-c['dims'][i]/2 for c in cub) for i in range(3)]; alhi = [max(c['pose'][i]+c['dims'][i]/2 for c in cub) for i in range(3)]
    sampling = dict(x=[-0.96, R_(alhi[0])], y=[R_(allo[1]), R_(alhi[1])], z=[R_(floor), R_(alhi[2])])
    focus = dict(x=[R_(allo[0]), R_(alhi[0])], y=[R_(allo[1]), R_(alhi[1])], z=[R_(floor), R_(alhi[2])])
    return dict(cuboids=cuboids, sampling=sampling, focus=focus, compartments=comps, viz_table=base['viz_table'], home_box=[lo, hi])


def text(d):
    f = lambda v: ', '.join(repr(R_(x)) for x in v)
    L = [f'# Square scene {NAME}: three easy cupboards (cabinet_easy: 2 x wide compartments, 0.10 m panels) around the robot at +x, +y, -y,',
         '# each centred on its axis and open towards the robot; one table slab under each. Written by jeffrey/easy_cabinet/make_square.py; do not edit by hand.',
         f'# Home cupboard (+x) outer box: x {R_(d["home_box"][0][0])}-{R_(d["home_box"][1][0])}, y {R_(d["home_box"][0][1])}-{R_(d["home_box"][1][1])}, z {R_(d["home_box"][0][2])}-{R_(d["home_box"][1][2])}; the others are its 90 deg turns.',
         '# No task orientation: no zones or task_axes. Speed is clearance only.', '', 'cuboids:']
    for c in d['cuboids']:
        L += [f'  - name: {c["name"]}', f'    pose: [{f(c["pose"])}]', f'    dims: [{f(c["dims"])}]']+([f'    color: [{", ".join(map(str, c["color"]))}]'] if 'color' in c else [])+['']
    v = d['viz_table']; L += ['viz_table:', f'  name: {v["name"]}', f'  pose: [{f(v["pose"])}]', f'  dims: [{f(v["dims"])}]', '', '# Grasp-point sampling box: the square around the robot, z from the table top to the cupboard top.', 'sampling:']
    L += [f'  {k}: [{f(d["sampling"][k])}]' for k in 'xyz']+['  rule: the three cupboards and the strip behind the robot (x >= -0.96); z = cupboard height']
    L += ['', '# The region that takes the cupboard\'s place in the coverage strata and the test regions: the bounding box of the three cupboards.', 'focus:']+[f'  {k}: [{f(d["focus"][k])}]' for k in 'xyz']
    L += ['', '# Grasp-point boxes of the nine compartment interiors (0.04 m past each opening).', 'compartments:']
    for c in d['compartments']: L += [f'  - name: {c["name"]}']+[f'    {k}: [{f(c[k])}]' for k in 'xyz']
    L += ['', '# Derivation (jeffrey/easy_cabinet/make_square.py); not read by the runtime.', 'variant:', '  base: cabinet_easy.yml', '  clearance_scale: 1.0', '  copies: [px, py, ny]',
          '  rule: three copies of the easy cupboard, lateral centre 0, open towards the robot at +x, +y (turned +90 deg) and -y (turned -90 deg); table slabs under each; focus = bounding box of the three', '']
    return '\n'.join(L)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); a = p.parse_args()
    d = derive(); t = text(d); doc = yaml.safe_load(t); path = SCENES/(NAME+'.yml')
    assert doc['cuboids'] == d['cuboids'] and {k: doc['sampling'][k] for k in 'xyz'} == d['sampling'] and doc['compartments'] == d['compartments'] and doc['focus'] == d['focus']
    # the three cupboards must not touch each other, and every cupboard must stand on a slab
    cups = {how: [c for c in d['cuboids'] if c['name'].endswith('_'+how) and not c['name'].startswith('table')] for how in ('px', 'py', 'ny')}
    box = lambda cs: [[min(c['pose'][i]-c['dims'][i]/2 for c in cs), max(c['pose'][i]+c['dims'][i]/2 for c in cs)] for i in range(3)]
    B = {k: box(v) for k, v in cups.items()}
    for a_, b_ in (('px', 'py'), ('px', 'ny'), ('py', 'ny')):
        assert any(B[a_][i][1] <= B[b_][i][0] or B[b_][i][1] <= B[a_][i][0] for i in range(2)), f'cupboards {a_} and {b_} overlap: {B[a_]} {B[b_]}'
    if a.check: assert path.read_text() == t, f'{path} differs'; print('ok'); return
    if path.exists() and path.read_text() != t: raise SystemExit(f'{path} exists and differs; frozen suites record its hash')
    path.write_text(t); print(path, dict(boxes=B, sampling=d['sampling'], focus=d['focus'], compartments=[c['name'] for c in d['compartments']]))


if __name__ == '__main__': main()
