"""Derive the easy cabinet, experiments/cabinet/scenes/cabinet_easy.yml: every compartment opening 2.0 x as wide (the width of
cabinet_w20, centre line y 0.15, compartment heights unchanged so the top compartment stays within reach) with every panel
5 x as thick (0.10 m), panels growing outward from the compartment interiors (jeffrey/joint_range/make_strong.py's rule).

    python3 jeffrey/easy_cabinet/make_easy.py           # writes the scene
    python3 jeffrey/easy_cabinet/make_easy.py --check   # derives it again and compares with the file
"""
import argparse, importlib.util
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]; SCENES = ROOT/'experiments/cabinet/scenes'
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
WIDE = load('ntrl_make_scenes', ROOT/'experiments/ntrl_cabinet/make_scenes.py'); STRONG = load('make_strong', ROOT/'jeffrey/joint_range/make_strong.py')
NAME, SW, SH, YC, K = 'cabinet_easy', 2.0, 1.0, 0.15, 5.0


def derive():
    base = yaml.safe_load((SCENES/'cabinet.yml').read_text()); b, wide = WIDE.derive(base, SW, SH, YC)
    wide_doc = dict(cuboids=wide['cuboids'], sampling=wide['sampling'], compartments=wide['compartments'], viz_table=base['viz_table'])
    d = STRONG.derive(wide_doc, K); return base, wide_doc, d


def text(base, d):
    f = lambda v: ', '.join(repr(STRONG.r(x)) for x in v); o, sp, cm = d['outer'], d['sampling'], d['compartments']
    L = [f'# Easy cabinet {NAME}: cabinet_w20 (compartment openings {SW} x as wide, {SH} x as tall, centre line y {YC}) with every panel',
         f'# {K:g} x as thick ({STRONG.r(d["thickness"])} m), panels growing outward from the unchanged compartment interiors. Written by',
         '# jeffrey/easy_cabinet/make_easy.py; do not edit by hand.',
         f'# Cupboard (open towards the robot, -X): outer box x {STRONG.r(d["opening"])}-{STRONG.r(d["back_out"])}, y {STRONG.r(d["yc"]-o[1]/2)}-{STRONG.r(d["yc"]+o[1]/2)}, z {sp["z"][0]}-{sp["z"][1]}.',
         '# Compartments: '+', '.join(f'{x["name"]} y {x["y"][0]}-{x["y"][1]} z {x["z"][0]}-{x["z"][1]}' for x in cm)+'.', '#',
         "# Sampling box for the grasp point: height = cupboard, width 1.5 x centred, depth 2.5 x from the back face but x >= 0.08.",
         '# No task orientation: no zones or task_axes. Speed is clearance only.', '', 'cuboids:']
    for c in d['cuboids']:
        L += [f'  - name: {c["name"]}', f'    pose: [{f(c["pose"])}]', f'    dims: [{f(c["dims"])}]']+([f'    color: [{", ".join(map(str, c["color"]))}]'] if 'color' in c else [])+['']
    v = base['viz_table']; L += ['viz_table:', f'  name: {v["name"]}', f'  pose: [{f(v["pose"])}]', f'  dims: [{f(v["dims"])}]', '', 'sampling:']
    L += [f'  {k}: [{f(sp[k])}]' for k in 'xyz']+['  rule: height = cupboard, width = 1.5 x, depth = 2.5 x from the back face, x >= 0.08']
    L += ['', '# Grasp-point boxes of the compartment interiors (0.04 m past the opening). Used by the test suites and the coverage strata.', 'compartments:']
    for c in cm: L += [f'  - name: {c["name"]}']+[f'    {k}: [{f(c[k])}]' for k in 'xyz']
    L += ['', '# Derivation (jeffrey/easy_cabinet/make_easy.py); not read by the runtime.', 'variant:', '  base: cabinet.yml', f'  clearance_scale: {SH}', f'  width_scale: {SW}', f'  height_scale: {SH}',
          f'  centre_y: {YC}', f'  panel_scale: {K:g}',
          '  rule: compartment openings width_scale x as wide and height_scale x as tall, cupboard centred on y = centre_y (cabinet_w20); then every panel panel_scale x as thick, growing outward from the interiors', '']
    return '\n'.join(L)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); a = p.parse_args()
    base, wide_doc, d = derive(); t = text(base, d); doc = yaml.safe_load(t); path = SCENES/(NAME+'.yml')
    assert doc['cuboids'] == d['cuboids'] and {k: doc['sampling'][k] for k in 'xyz'} == d['sampling'] and doc['compartments'] == d['compartments']
    assert doc['compartments'][0]['y'] == wide_doc['compartments'][0]['y'], 'interiors changed'
    if a.check: assert path.read_text() == t, f'{path} differs'; print('ok'); return
    if path.exists() and path.read_text() != t: raise SystemExit(f'{path} exists and differs; frozen suites record its hash')
    path.write_text(t); print(path, dict(thickness=d['thickness'], outer=[round(x, 3) for x in d['outer']], sampling=d['sampling'], compartments=d['compartments']))


if __name__ == '__main__': main()
