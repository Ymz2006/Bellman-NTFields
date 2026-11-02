"""Derive larger cabinet scenes (wider and taller compartments) from experiments/cabinet/scenes/cabinet.yml.

    python3 experiments/ntrl_cabinet/make_scenes.py           # writes experiments/cabinet/scenes/cabinet_<name>.yml
    python3 experiments/ntrl_cabinet/make_scenes.py --check   # derives them again and compares with the files

Same derivation as experiments/cabinet/make_scenes.py (panels, floor, back face, depth unchanged; the
sampling box and compartment boxes follow the task rule), with two changes so that much larger cupboards
stay reachable by the UR5e at the origin:
- separate scales for the compartment width (sw) and height (sh); a 2x taller cupboard would put the top
  compartment out of reach, while the ntrl-demo paper cabinet's compartments are ~0.9 m wide and ~0.35 m tall;
- the cupboard's centre line moves to y = yc (cabinet.yml: 0.33) so the wider cupboard grows towards the
  robot's centre line instead of out of reach.
"""
import argparse,copy,importlib.util
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('cabinet_make_scenes',ROOT/'experiments/cabinet/make_scenes.py')
BASE_SCENES=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE_SCENES)
SCENES=ROOT/'experiments/cabinet/scenes'
# name: (width scale, height scale, centre line y)
VARIANTS={'cabinet_w15':(1.5,1.2,.24),'cabinet_w20':(2.0,1.25,.15),'cabinet_w25':(2.5,1.3,.06)}


def derive(base,sw,sh,yc):
    b=copy.deepcopy(base);c={x['name']:x for x in b['cuboids']}
    c['shelf_top']['dims'][1]*=sw/sh   # derive() takes the width as s x the top panel's width
    for name in ['shelf_back','shelf_side_inner','shelf_side_outer','shelf_top','shelf_board_low','shelf_board_high']:c[name]['pose'][1]=yc
    d=BASE_SCENES.derive(b,sh)
    for cub in d['cuboids']:
        if cub['name']=='shelf_top':cub['dims'][1]=BASE_SCENES.r(cub['dims'][1])
    return b,d


def body(name,sw,sh,yc,base,b,d):
    text=BASE_SCENES.text(name,sh,b,d).split('\n')
    head=[f'# Larger cabinet {name}: scenes/cabinet.yml with every compartment opening {sw} x as wide and each compartment',
          f'# {sh} x as tall, centre line moved to y {yc}. Written by experiments/ntrl_cabinet/make_scenes.py; do not edit by hand.']
    text=[('# Unchanged from cabinet.yml: table, panel thickness, floor, back face, depth.' if l.startswith('# Unchanged') else l) for l in text]
    text=head+[l for l in text[2:] if not l.startswith('variant:') and not l.startswith('  base:') and not l.startswith('  clearance_scale:') and not l.startswith('  rule:')]
    while text and text[-1]=='':text.pop()
    text+=['variant:','  base: cabinet.yml',f'  clearance_scale: {sh}',f'  width_scale: {sw}',f'  height_scale: {sh}',f'  centre_y: {yc}',
           '  rule: each compartment opening width_scale x as wide and each compartment height_scale x as tall, cupboard centred on y = centre_y; panels, floor, back face and depth unchanged','']
    return '\n'.join(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    base=yaml.safe_load((SCENES/'cabinet.yml').read_text())
    _,same=derive(base,1.,1.,.33)
    assert same['cuboids']==base['cuboids'] and same['sampling']==base['sampling'] and same['compartments']==base['compartments'],'identity does not reproduce cabinet.yml'
    for name,(sw,sh,yc) in VARIANTS.items():
        b,d=derive(base,sw,sh,yc);text=body(name,sw,sh,yc,base,b,d);doc=yaml.safe_load(text);path=SCENES/(name+'.yml')
        assert doc['cuboids']==d['cuboids'] and doc['sampling']==d['sampling'] and doc['compartments']==d['compartments']
        if a.check:assert path.read_text()==text,f'{path} differs from its derivation'
        elif path.exists() and path.read_text()!=text:raise SystemExit(f'{path} exists and differs; frozen suites record its hash')
        else:path.write_text(text)
        print(name,dict(width=round(d['width'],4),heights=[round(h,4) for h in d['heights']],outer=[round(x,4) for x in d['outer']],sampling=d['sampling']))


if __name__=='__main__':main()
