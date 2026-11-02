"""Derive the strong cabinet, experiments/cabinet/scenes/cabinet_strong.yml, from experiments/cabinet/scenes/cabinet.yml.

    python3 jeffrey/joint_range/make_strong.py           # writes the scene
    python3 jeffrey/joint_range/make_strong.py --check   # derives it again and compares with the file

Every panel of the cupboard (back, two sides, top, the two shelf boards) is 5x as thick: 0.02 -> 0.10 m. Every
compartment keeps its inside size (0.30 deep, 0.36 wide; 0.225, 0.21 and 0.205 m tall), so the panels grow outward:
- the opening (x 0.56), the floor (table top), the centre line (y 0.33) and the bottom compartment stay where they are;
  the back panel's inner face stays at x 0.86 and its back face moves out to x 0.96, the sides grow to y 0.05 and 0.61;
- the compartments stack from the floor as before (compartment, board, compartment, board, compartment, top panel),
  so with 0.10 m boards the middle compartment moves up 0.08 m and the top compartment 0.16 m; the cupboard is
  0.40 x 0.56 x 0.94 m instead of 0.32 x 0.40 x 0.70 m.
Sampling box (test area) by the task rule of experiments/cabinet/make_scenes.py from the new outer box (height =
cupboard, width 1.5 x centred, depth 2.5 x from the back face), except that it starts no closer to the robot than
cabinet.yml's x 0.08 (2.5 x 0.40 would reach x -0.04, over the robot base). Compartment boxes: the interiors from 0.04 m
past the opening, as in cabinet.yml. With scale 1 the derivation reproduces cabinet.yml (checked).
"""
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT/'experiments/cabinet/scenes'
SCALE, NAME = 5.0, 'cabinet_strong'
WIDTH, DEPTH, INSET, X_MIN = 1.5, 2.5, .04, .08


def r(v): return round(float(v), 6)


def derive(base, k):
    c = {x['name']: x for x in base['cuboids']}; table, back, side, top = c['table'], c['shelf_back'], c['shelf_side_inner'], c['shelf_top']
    t0 = back['dims'][0]; t = k*t0; floor = table['pose'][2]+table['dims'][2]/2; yc = back['pose'][1]
    opening = side['pose'][0]-side['dims'][0]/2; back_in = back['pose'][0]-t0/2; back_out = back_in+t; depth = back_in-opening
    width = top['dims'][1]; heights = [x['z'][1]-x['z'][0] for x in base['compartments']]
    z = [floor]; levels = {}
    for name, h, board in zip([x['name'] for x in base['compartments']], heights, ['shelf_board_low', 'shelf_board_high', 'shelf_top']):
        levels[name] = (z[-1], z[-1]+h); levels[board] = (z[-1]+h, z[-1]+h+t); z.append(z[-1]+h+t)
    height = z[-1]-floor; mid = (opening+back_in)/2; outer = width+2*t
    pose = lambda x, y, zz, src: [r(x), r(y), r(zz)]+src['pose'][3:]
    cuboids = [table, dict(name='shelf_back', pose=pose(back_in+t/2, yc, floor+height/2, back), dims=[r(t), r(outer), r(height)], color=back['color'])]
    for name, y in [('shelf_side_inner', yc-width/2-t/2), ('shelf_side_outer', yc+width/2+t/2)]:
        cuboids.append(dict(name=name, pose=pose(mid, y, floor+height/2, c[name]), dims=[r(depth), r(t), r(height)], color=c[name]['color']))
    for name in ['shelf_top', 'shelf_board_low', 'shelf_board_high']:
        lo, hi = levels[name]; cuboids.append(dict(name=name, pose=pose(mid, yc, (lo+hi)/2, c[name]), dims=[r(depth), r(width), r(t)], color=c[name]['color']))
    sampling = dict(x=[r(max(X_MIN, back_out-DEPTH*(back_out-opening))), r(back_out)], y=[r(yc-WIDTH*outer/2), r(yc+WIDTH*outer/2)], z=[r(floor), r(floor+height)])
    compartments = [dict(name=x['name'], x=[r(opening+INSET), r(back_in)], y=[r(yc-width/2), r(yc+width/2)], z=[r(v) for v in levels[x['name']]]) for x in base['compartments']]
    return dict(cuboids=cuboids, sampling=sampling, compartments=compartments, thickness=t, outer=[back_out-opening, outer, height], yc=yc, opening=opening, back_out=back_out)


def text(base, d):
    f = lambda v: ', '.join(repr(r(x)) for x in v)
    o, sp, cm = d['outer'], d['sampling'], d['compartments']
    L = [f'# Strong cabinet {NAME}: scenes/cabinet.yml with every panel {SCALE:g} x as thick ({r(d["thickness"])} m) and every compartment',
         '# the same size inside. Written by jeffrey/joint_range/make_strong.py; do not edit by hand.',
         f'# Cupboard (open towards the robot, -X): outer box x {r(d["opening"])}-{r(d["back_out"])}, y {r(d["yc"]-o[1]/2)}-{r(d["yc"]+o[1]/2)}, z {sp["z"][0]}-{sp["z"][1]},',
         f'# {r(d["thickness"])} m panels. Three compartments 0.3 deep x 0.36 wide (y {cm[0]["y"][0]}-{cm[0]["y"][1]}): '
         + ', '.join(f'{x["name"]} z {x["z"][0]}-{x["z"][1]}' for x in cm)+'.',
         '# Unchanged from cabinet.yml: table, opening, floor, centre line, every compartment interior size; the bottom',
         '# compartment is in the same place, the middle one 0.08 m and the top one 0.16 m higher (thicker boards below them).', '#',
         "# Sampling box for the grasp point, from the cupboard's outer box:",
         f'#   height = cupboard height = {r(o[2])}  z {sp["z"][0]}-{sp["z"][1]}',
         f'#   width  = 1.5 x {r(o[1])} = {r(1.5*o[1])}  y {sp["y"][0]}-{sp["y"][1]} (centred on the cupboard)',
         f'#   depth  = 2.5 x {r(o[0])} from the back face, but not closer to the robot than x {X_MIN}  x {sp["x"][0]}-{sp["x"][1]}',
         '#', '# No task orientation: no zones or task_axes. Speed is clearance only.', '', 'cuboids:']
    for c in d['cuboids']:
        L += [f'  - name: {c["name"]}', f'    pose: [{f(c["pose"])}]', f'    dims: [{f(c["dims"])}]']+([f'    color: [{", ".join(map(str, c["color"]))}]'] if 'color' in c else [])+['']
    v = base['viz_table']; L += ['viz_table:', f'  name: {v["name"]}', f'  pose: [{f(v["pose"])}]', f'  dims: [{f(v["dims"])}]', '', 'sampling:']
    L += [f'  {k}: [{f(sp[k])}]' for k in 'xyz']+[f'  rule: height = cupboard, width = 1.5 x, depth = 2.5 x from the back face, x >= {X_MIN}']
    L += ['', '# Grasp-point boxes of the compartment interiors (0.04 m past the opening). Used by the test suites and the coverage strata.', 'compartments:']
    for c in cm: L += [f'  - name: {c["name"]}']+[f'    {k}: [{f(c[k])}]' for k in 'xyz']
    L += ['', '# Derivation (jeffrey/joint_range/make_strong.py); not read by the runtime.', 'variant:', '  base: cabinet.yml', '  clearance_scale: 1.0', f'  panel_scale: {SCALE:g}',
          f'  rule: every panel panel_scale x as thick, every compartment interior the same size; panels grow outward from the interiors (opening, floor, centre line fixed; compartments stacked from the floor)', '']
    return '\n'.join(L)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); a = p.parse_args()
    base = yaml.safe_load((SCENES/'cabinet.yml').read_text()); same = derive(base, 1.)
    assert same['cuboids'] == base['cuboids'] and same['sampling'] == base['sampling'] and same['compartments'] == base['compartments'], 'scale 1 does not reproduce cabinet.yml'
    d = derive(base, SCALE); body = text(base, d); doc = yaml.safe_load(body); path = SCENES/(NAME+'.yml')
    assert doc['cuboids'] == d['cuboids'] and {k: doc['sampling'][k] for k in 'xyz'} == d['sampling'] and doc['compartments'] == d['compartments']
    if a.check: assert path.read_text() == body, f'{path} differs from its derivation'
    elif path.exists() and path.read_text() != body: raise SystemExit(f'{path} exists and differs; frozen suites record its hash')
    else: path.write_text(body)
    print(NAME, dict(thickness=d['thickness'], outer=[r(x) for x in d['outer']], sampling=d['sampling'], compartments={c['name']: c['z'] for c in d['compartments']}))


if __name__ == '__main__': main()
