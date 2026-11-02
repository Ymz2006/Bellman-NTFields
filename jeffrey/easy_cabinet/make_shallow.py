"""Derive the shallow wide cabinet, experiments/cabinet/scenes/cabinet_shallow.yml, from cabinet_w20.yml: the same wide
cupboard (0.72 m compartments, 0.02 m panels, centre line y 0.15) with the compartments 0.20 m deep instead of 0.30 m.
The back panel stays at x 0.86-0.88; the sides, top and boards shorten from the opening side, so the opening moves from
x 0.56 to x 0.66 (less reaching in). Sampling box by the task rule (depth 2.5 x the outer depth from the back face).

    python3 jeffrey/easy_cabinet/make_shallow.py [--check]
"""
import argparse, copy
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]; SCENES = ROOT/'experiments/cabinet/scenes'; NAME, DEPTH, INSET, RULE = 'cabinet_shallow', 0.20, 0.04, 2.5
r = lambda v: round(float(v), 6)


def derive():
    base = yaml.safe_load((SCENES/'cabinet_w20.yml').read_text()); d = copy.deepcopy(base); c = {x['name']: x for x in d['cuboids']}
    back = c['shelf_back']; t = back['dims'][0]; back_in = back['pose'][0]-t/2; back_out = back_in+t; opening = back_in-DEPTH; mid = (opening+back_in)/2
    for name in ('shelf_side_inner', 'shelf_side_outer', 'shelf_top', 'shelf_board_low', 'shelf_board_high'):
        c[name]['pose'][0] = r(mid); c[name]['dims'][0] = r(DEPTH)
    for comp in d['compartments']: comp['x'] = [r(opening+INSET), r(back_in)]
    d['sampling']['x'] = [r(back_out-RULE*(back_out-opening)), r(back_out)]
    d['variant'] = dict(base='cabinet_w20.yml', clearance_scale=base['variant']['clearance_scale'], width_scale=base['variant']['width_scale'], height_scale=base['variant']['height_scale'],
                        centre_y=base['variant']['centre_y'], depth=DEPTH, rule=base['variant']['rule']+f'; compartments {DEPTH} m deep (opening moved towards the back panel)')
    return d


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); a = p.parse_args(); d = derive(); path = SCENES/(NAME+'.yml')
    head = (f'# Shallow wide cabinet {NAME}: cabinet_w20 with the compartments {DEPTH} m deep (opening at x {d["compartments"][0]["x"][0]-INSET:.2f}).\n'
            '# Written by jeffrey/easy_cabinet/make_shallow.py; do not edit by hand.\n')
    t = head+yaml.safe_dump({k: d[k] for k in ('cuboids', 'viz_table', 'sampling', 'compartments', 'variant')}, sort_keys=False)
    if a.check: assert path.read_text() == t; print('ok'); return
    if path.exists() and path.read_text() != t: raise SystemExit(f'{path} exists and differs')
    path.write_text(t); print(path, d['sampling'], d['compartments'][0])


if __name__ == '__main__': main()
