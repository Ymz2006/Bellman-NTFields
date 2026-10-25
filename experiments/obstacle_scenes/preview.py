"""Viser preview of the obstacle scenes: geometry, target regions, example valid poses and
which frozen-600 query endpoints each scene invalidates. Read-only; no planning.

    ./docker-python experiments/obstacle_scenes/preview.py --data <dir with preview.npz> --port 8903
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
import numpy as np
import trimesh
import viser
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'gui'))
import helpers as helper

SCENES=('platforms','shelf','clutter')
TITLES={'platforms':'Raised platforms under both task regions','shelf':'Standing cupboard with shelves (goals inside)','clutter':'Cluttered counter (one object per task region)'}
TARGET_COLORS=((40,200,220),(250,120,200),(150,220,60))
# Cameras that look into each scene's obstacles; the shelf opens towards the robot (-x).
CAMERAS={'platforms':((1.55,-1.35,1.05),(0.55,0.0,0.1)),'shelf':((-0.05,0.62,0.8),(0.72,0.3,0.22)),'clutter':((1.45,-1.25,1.1),(0.55,0.0,0.1))}


def box_edges(x,y,z):
    corners=np.array([[a,b,c] for a in x for b in y for c in z],dtype=np.float32)
    pairs=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]
    return np.stack([corners[[i,j]] for i,j in pairs])


def describe(name,doc,summary):
    lines=[f'## {TITLES[name]}','','|Box|x (m)|y (m)|z (m)|','|---|---|---|---|']
    for c in doc['cuboids'][1:]:
        (px,py,pz),(dx,dy,dz)=c['pose'][:3],c['dims']
        lines.append(f"|{c['name']}|{px-dx/2:.2f}–{px+dx/2:.2f}|{py-dy/2:+.2f}–{py+dy/2:+.2f}|{pz-dz/2:.3f}–{pz+dz/2:.3f}|")
    s=summary['scenes'][name]
    lines+=['',f"**Frozen 600 queries with both endpoints still valid: {s['frozen_endpoint_valid_queries']}/600** "
            f"({s['frozen_valid_endpoints']}/1200 endpoints). Red dots: endpoints this scene blocks.",'',
            '**Target regions** (cyan wireframes) for the scene-specific queries; valid poses found by uniform joint sampling:']
    lines+=[f"- {t}: {v['valid_samples']} valid poses" for t,v in s['targets'].items()]
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True,help='directory holding preview.npz and preview.json from design.py')
    p.add_argument('--port',type=int,default=8903);p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--scene',choices=SCENES,default='shelf');p.add_argument('--target',help='initial target region');p.add_argument('--example',type=int,default=0)
    p.add_argument('--frozen',action='store_true',help='start with frozen-600 endpoints shown');a=p.parse_args()
    data=np.load(a.data/'preview.npz');summary=json.loads((a.data/'preview.json').read_text())
    docs={name:yaml.safe_load((HERE/'scenes'/(name+'.yml')).read_text()) for name in SCENES};regions=yaml.safe_load((HERE/'targets.yml').read_text())
    server=viser.ViserServer(host=a.host,port=a.port,label='Obstacle scene preview')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/grid',width=2,height=2,cell_size=.1,plane='xy')
    server.scene.add_light_ambient('/ambient',intensity=.7)
    server.scene.add_light_directional('/key',intensity=1.2,wxyz=(.85,.35,.35,.15))
    for table in helper.load_viz_tables_yaml(str(HERE/'scenes/shelf.yml')):helper.add_viz_table_to_viser(server,table)
    urdf=helper.load_urdf(str(ROOT/'runtime/assets/runtime/robot/ur5e_robotiq_2f_140.urdf'))
    robot=helper.add_robot_scaled(server,urdf)
    gripper=server.scene.add_frame('/gripper_frame',axes_length=.09,axes_radius=.004,origin_radius=.008)

    nodes={}
    for name,doc in docs.items():
        handles=[]
        for c in doc['cuboids'][1:]:
            handles.append(server.scene.add_box(f'/obstacles/{name}/{c["name"]}',dimensions=tuple(c['dims']),color=tuple(c.get('color',(235,72,42))),
                                                opacity=.92,position=tuple(c['pose'][:3]),wxyz=tuple(c['pose'][3:])))
        groups={}
        for c in doc['cuboids'][1:]:
            group='cupboard' if c['name'].startswith('shelf_') else c['name'];(px,py,pz),(dx,dy,dz)=c['pose'][:3],c['dims']
            top=pz+dz/2;groups[group]=(px,py,max(top,groups.get(group,(0,0,-1))[2]))
        for group,(px,py,top) in groups.items():
            handles.append(server.scene.add_label(f'/obstacles/{name}/{group}_label',text=group,position=(px,py,top+.04)))
        targets=[]
        for i,t in enumerate(regions.get(name,[])):
            color=TARGET_COLORS[i%len(TARGET_COLORS)]
            targets.append(server.scene.add_line_segments(f'/targets/{name}/{t["name"]}/box',points=box_edges(t['x'],t['y'],t['z']),colors=(40,200,220),line_width=2))
            key=f'{name}/{t["name"]}/cloud'
            if key in data.files:
                targets.append(server.scene.add_point_cloud(f'/targets/{name}/{t["name"]}/cloud',points=data[key][:1500],colors=color,point_size=.005))
        ok=data[name+'/frozen_ok'];ee=data['frozen_ee']
        frozen=[server.scene.add_point_cloud(f'/frozen/{name}/valid',points=ee[ok],colors=(60,200,90),point_size=.007),
                server.scene.add_point_cloud(f'/frozen/{name}/blocked',points=ee[~ok],colors=(230,40,40),point_size=.011)]
        nodes[name]=dict(obstacles=handles,targets=targets,frozen=frozen)

    server.gui.add_markdown('# Obstacle scenes\nSame table, task regions, 20° orientation rules and 600 frozen queries as wall40/wall80; only boxes are added. '
                            'The robot shows example poses that pass the strict endpoint check (whole-arm clearance ≥ 5 mm, orientation rule of its region).')
    scene=server.gui.add_dropdown('Scene',options=SCENES,initial_value=a.scene)
    target=server.gui.add_dropdown('Example poses in',options=('…',),initial_value='…')
    example=server.gui.add_slider('Example',min=0,max=7,step=1,initial_value=0)
    play=server.gui.add_checkbox('Cycle examples',initial_value=False)
    show_targets=server.gui.add_checkbox('Show target regions and valid poses',initial_value=True)
    show_frozen=server.gui.add_checkbox('Show frozen-600 endpoints (green valid, red blocked)',initial_value=a.frozen)
    pose_info=server.gui.add_markdown('')
    info=server.gui.add_markdown('')
    lock=threading.RLock()

    def place():
        with lock:
            key=f'{scene.value}/{target.value}/examples_q'
            if key not in data.files:return
            qs=data[key];i=min(int(example.value),len(qs)-1);helper.update_robot(urdf,robot,qs[i])
            transform=helper.get_link_transform(urdf,'grasp_frame');gripper.position=tuple(transform[:3,3])
            gripper.wxyz=tuple(trimesh.transformations.quaternion_from_matrix(transform))
            x,y,z=data[f'{scene.value}/{target.value}/examples_ee'][i]
            pose_info.content=f'**{target.value} · example {i+1}/{len(qs)}** · grasp point x={x:.3f}, y={y:+.3f}, z={z:.3f} m'

    def visibility():
        for name,group in nodes.items():
            for h in group['obstacles']:h.visible=name==scene.value
            for h in group['targets']:h.visible=name==scene.value and show_targets.value
            for h in group['frozen']:h.visible=name==scene.value and show_frozen.value

    def switch(_=None):
        with lock:
            doc=docs[scene.value];names=tuple(t['name'] for t in regions.get(scene.value,[]))
            target.options=names;target.value=names[0];example.value=0
            helper.add_corner_table_zones_to_viser(server,dict(doc['zones'],draw_routing_map=False),helper.load_viz_tables_yaml(str(HERE/'scenes'/(scene.value+'.yml'))),doc['cuboids'])
            visibility();info.content=describe(scene.value,doc,summary);place()
            position,look=CAMERAS[scene.value]
            for client in server.get_clients().values():
                client.camera.position=position;client.camera.look_at=look;client.camera.up=(0,0,1)

    scene.on_update(switch)
    target.on_update(lambda _:(setattr(example,'value',0),place()))
    example.on_update(lambda _:place())
    show_targets.on_update(lambda _:visibility());show_frozen.on_update(lambda _:visibility())

    @server.on_client_connect
    def connected(client):
        position,look=CAMERAS[scene.value]
        client.camera.position=position;client.camera.look_at=look;client.camera.up=(0,0,1)
    switch()
    if a.target in target.options:target.value=a.target
    example.value=a.example;place()
    print(f'READY http://localhost:{a.port}',flush=True)
    while True:
        time.sleep(1.5)
        if play.value:
            with lock:example.value=(int(example.value)+1)%8


if __name__=='__main__':main()
