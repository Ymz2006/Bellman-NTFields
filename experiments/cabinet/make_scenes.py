"""Derive the larger-clearance cabinet scenes from scenes/cabinet.yml.

    python3 experiments/cabinet/make_scenes.py           # writes scenes/cabinet_plus10.yml and scenes/cabinet_plus20.yml
    python3 experiments/cabinet/make_scenes.py --check   # derives them again and compares with the files

Clearance scale s (1.1 for plus10, 1.2 for plus20): every compartment opening is s times as wide and
each compartment s times as tall as in the cabinet scene (inside 0.36 m wide; 0.225, 0.21 and 0.205 m
tall), so the arm has 10% / 20% more room in every compartment. Unchanged: the 0.02 m panels, the floor
(the table top), the centre line y 0.33, the back face at x 0.88 and the depth (0.30 m inside): depth adds
reach, not room around the arm. The sampling box follows the task rule from the new outer box (height =
cupboard, width 1.5 x centred, depth 2.5 x from the back face); the compartment boxes are the new
interiors from 0.04 m past the opening. With s = 1 the derivation reproduces scenes/cabinet.yml (checked).
"""
import argparse
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
BASE=HERE/'scenes/cabinet.yml'
VARIANTS={'cabinet_plus10':1.1,'cabinet_plus20':1.2}
WIDTH,DEPTH=1.5,2.5
INSET=.04


def r(v):return round(float(v),6)


def derive(base,s):
    """Cuboids, sampling box and compartments of the cupboard with clearance scale s."""
    c={x['name']:x for x in base['cuboids']};table,back,side,top=c['table'],c['shelf_back'],c['shelf_side_inner'],c['shelf_top']
    t=back['dims'][0];floor=table['pose'][2]+table['dims'][2]/2;yc=back['pose'][1]
    back_out=back['pose'][0]+t/2;back_in=back_out-t;opening=side['pose'][0]-side['dims'][0]/2;depth=back_in-opening
    width=s*top['dims'][1];heights=[s*(x['z'][1]-x['z'][0]) for x in base['compartments']]
    # Stack from the floor: compartment, board, compartment, board, compartment, top panel.
    z=[floor];levels={}
    for name,h,board in zip([x['name'] for x in base['compartments']],heights,['shelf_board_low','shelf_board_high','shelf_top']):
        levels[name]=(z[-1],z[-1]+h);levels[board]=(z[-1]+h,z[-1]+h+t);z.append(z[-1]+h+t)
    height=z[-1]-floor;mid=(opening+back_in)/2;outer=width+2*t
    pose=lambda x,y,zz,src:[r(x),r(y),r(zz)]+src['pose'][3:]
    cuboids=[table,dict(name='shelf_back',pose=pose(back_out-t/2,yc,floor+height/2,back),dims=[r(t),r(outer),r(height)],color=back['color'])]
    for name,y in [('shelf_side_inner',yc-width/2-t/2),('shelf_side_outer',yc+width/2+t/2)]:
        cuboids.append(dict(name=name,pose=pose(mid,y,floor+height/2,c[name]),dims=[r(depth),r(t),r(height)],color=c[name]['color']))
    for name in ['shelf_top','shelf_board_low','shelf_board_high']:
        lo,hi=levels[name];cuboids.append(dict(name=name,pose=pose(mid,yc,(lo+hi)/2,c[name]),dims=[r(depth),r(width),r(t)],color=c[name]['color']))
    sampling=dict(x=[r(back_out-DEPTH*(back_out-opening)),r(back_out)],y=[r(yc-WIDTH*outer/2),r(yc+WIDTH*outer/2)],z=[r(floor),r(floor+height)])
    compartments=[dict(name=x['name'],x=[r(opening+INSET),r(back_in)],y=[r(yc-width/2),r(yc+width/2)],z=[r(v) for v in levels[x['name']]]) for x in base['compartments']]
    return dict(cuboids=cuboids,sampling=sampling,compartments=compartments,width=width,heights=heights,outer=[back_out-opening,outer,height],yc=yc)


def text(name,s,base,d):
    f=lambda v:', '.join(repr(r(x)) for x in v)
    w=d['width'];o=d['outer'];sp=d['sampling'];h=d['heights']
    L=[f'# Cabinet task with {round(100*(s-1))}% more clearance: scenes/cabinet.yml with every compartment opening',
       f'# {s} x as wide and each compartment {s} x as tall. Written by make_scenes.py; do not edit by hand.',
       f'# Cupboard (open towards the robot, -X): outer box x {r(sp["x"][1]-o[0])}-{sp["x"][1]}, y {r(d["yc"]-o[1]/2)}-{r(d["yc"]+o[1]/2)}, z {sp["z"][0]}-{sp["z"][1]},',
       f'# 0.02 m panels. Three compartments {r(o[0]-0.02)} deep x {r(w)} wide (y {d["compartments"][0]["y"][0]}-{d["compartments"][0]["y"][1]}), '
       +', '.join(f'{x["name"]} {r(hh)} tall' for x,hh in zip(d['compartments'],h))+'.',
       '# Unchanged from cabinet.yml: table, panel thickness, floor, centre line, back face, depth.','#',
       '# Sampling box for the grasp point, from the cupboard\'s outer box:',
       f'#   height = cupboard height = {r(o[2])}  z {sp["z"][0]}-{sp["z"][1]} (FK sampler: table top + --height {r(sp["z"][1]-sp["z"][0])})',
       f'#   width  = 1.5 x {r(o[1])} = {r(1.5*o[1])}  y {sp["y"][0]}-{sp["y"][1]} (centred on the cupboard)',
       f'#   depth  = 2.5 x {r(o[0])} = {r(2.5*o[0])}  x {sp["x"][0]}-{sp["x"][1]} (back face of the cupboard towards the robot)',
       '#','# No task orientation: no zones or task_axes. Speed is clearance only.','','cuboids:']
    for c in d['cuboids']:
        L+=[f'  - name: {c["name"]}',f'    pose: [{f(c["pose"])}]',f'    dims: [{f(c["dims"])}]']+([f'    color: [{", ".join(map(str,c["color"]))}]'] if 'color' in c else [])+['']
    v=base['viz_table'];L+=['viz_table:',f'  name: {v["name"]}',f'  pose: [{f(v["pose"])}]',f'  dims: [{f(v["dims"])}]','',
       '# The frozen FK sampler reads x and y; z is recorded for the other stages and checked against it.','sampling:']
    L+=[f'  {k}: [{f(sp[k])}]' for k in 'xyz']
    L+=['','# Grasp-point boxes of the compartment interiors (0.04 m past the opening). Used by the test suites and the coverage strata.','compartments:']
    for c in d['compartments']:L+=[f'  - name: {c["name"]}']+[f'    {k}: [{f(c[k])}]' for k in 'xyz']
    L+=['','# Derivation (make_scenes.py); not read by the runtime.','variant:','  base: cabinet.yml',f'  clearance_scale: {s}',
        '  rule: each compartment opening clearance_scale x as wide and each compartment clearance_scale x as tall; panels, floor, centre line, back face and depth unchanged','']
    return '\n'.join(L)


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    base=yaml.safe_load(BASE.read_text());same=derive(base,1.)
    assert same['cuboids']==base['cuboids'] and same['sampling']==base['sampling'] and same['compartments']==base['compartments'],'s = 1 does not reproduce cabinet.yml'
    for name,s in VARIANTS.items():
        d=derive(base,s);body=text(name,s,base,d);doc=yaml.safe_load(body);path=HERE/'scenes'/(name+'.yml')
        assert doc['cuboids']==d['cuboids'] and doc['sampling']==d['sampling'] and doc['compartments']==d['compartments']
        if a.check:assert path.read_text()==body,f'{path} differs from its derivation'
        elif path.exists() and path.read_text()!=body:raise SystemExit(f'{path} exists and differs; the frozen suites record its hash')
        else:path.write_text(body)
        print(name,s,'sampling',d['sampling'],'outer',[r(x) for x in d['outer']])


if __name__=='__main__':main()
