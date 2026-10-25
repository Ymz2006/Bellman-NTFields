"""Read-only GUI replay of frozen Bellman600 and reference trajectories."""
import argparse
import importlib.util
import json
import threading
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import trimesh
import viser
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RELEASE=ROOT/'runtime'
EVAL=ROOT/json.loads((ROOT/'baseline.json').read_text())['evaluation']
import helpers as helper
METHODS=('Bellman + MPPI','Bellman RK4')
COLORS=((55,185,255),(185,105,255))
WALL=False
PLANNER="mppi"
TRANSFER=ROOT/"results/registered_20260930T003519Z-f50d55a8/evaluation"
# Without the original transfer run, its paths equal the no-wall baseline paths and its
# wall verdicts are the baseline verdicts with the recorded wall differences applied.
TRANSFER_EVIDENCE=ROOT/'evidence/wall25/comparison.json'


@lru_cache(maxsize=64)
def load(task,method,seed):
    if WALL:
        folder=EVAL if method==METHODS[0] else TRANSFER
        label=f'bellman_mppi_{task}_seed{seed}' if PLANNER=='mppi' else f'bellman_rk4_{task}'
        if method!=METHODS[0] and not TRANSFER.exists():
            folder=ROOT/json.loads((ROOT/'baseline.json').read_text())['evaluation']
    elif method==METHODS[0]: folder=EVAL;label=f'bellman_mppi_{task}_seed{seed}'
    elif method==METHODS[1]:folder=EVAL;label=f'bellman_rk4_{task}'
    with np.load(folder/(label+'.npz'),allow_pickle=True) as z:
        paths=[np.asarray(q,np.float32) for q in z['trajectories']]
    scores=json.loads((folder/(label+'.score.json')).read_text())[label]['per_query']
    if WALL and method!=METHODS[0] and not TRANSFER.exists():
        scores={k:dict(v) for k,v in scores.items()}
        for d in json.loads(TRANSFER_EVIDENCE.read_text())['comparisons'][PLANNER]['differences']:
            if d['label']==label:scores[str(d['query_id'])]=dict(success=d['new_success'],reasons=d['reasons'])
    return paths,scores


def success_total(method,seeds=None):
    tasks=[item['task'] for item in json.loads((RELEASE/'tests/suite_manifest.json').read_text())['suites']]
    seeds=seeds or ((0,1,2) if PLANNER=='mppi' else (0,))
    runs=[load(task,method,seed)[1] for task in tasks for seed in seeds]
    return sum(v['success'] for scores in runs for v in scores.values()),sum(len(scores) for scores in runs)


TILT_DEG=20.0
AXIS_COLORS={'x':(255,90,90),'y':(90,210,110),'z':(90,150,255)}
AXIS_COLOR_NAMES={'x':'Red','y':'Green','z':'Blue'}
RULE_ZONES={'cup':'cupping','pour':'pouring'}
WORLD_NAMES={(0,0,1):'up',(0,0,-1):'straight down',(1,0,0):'forward (world +X, away from base)',
             (-1,0,0):'back (world −X)',(0,1,0):'left (world +Y)',(0,-1,0):'right (world −Y)'}


def gripper_rules(axes):
    """Per task: which gripper axis is constrained and where its positive end must point."""
    rules={}
    for name,spec in axes.items():
        local=np.asarray(spec['local_axis'],float);i=int(np.argmax(np.abs(local)))
        assert np.allclose(np.abs(local),np.eye(3)[i]),'viewer expects a coordinate axis as local_axis'
        world=np.asarray(spec['world_axis'],float);world/=np.linalg.norm(world)
        rules[name]=dict(local=local,world=world,axis='xyz'[i],target=np.sign(local[i])*world)
    return rules


def tilt_deg(R,rule):
    # Same definition as runtime ur5e_task_axes.tilt_deg_from_quat.
    return float(np.degrees(np.arccos(np.clip((R@rule['local'])@rule['world'],-1.0,1.0))))


def zone_of(x,y,zones):
    # Mirrors runtime zone_router.corner_masks_torch with assign_outside=True.
    def inside(b):return b[0]<=x<=b[1] and b[2]<=y<=b[3]
    def distance(b):return float(np.hypot(max(b[0]-x,x-b[1],0.0),max(b[2]-y,y-b[3],0.0)))
    cup,pour=inside(zones['cup_region']),inside(zones['pour_region'])
    if (cup and pour) or ('overlap_region' in zones and inside(zones['overlap_region'])):return 'transition'
    if cup or pour:return 'cupping' if cup else 'pouring'
    return 'cupping' if distance(zones['cup_region'])<=distance(zones['pour_region']) else 'pouring'


def tolerance_cone(target,length=.12,tilt=TILT_DEG,count=32):
    """Line segments of a cone around `target`, anchored at the origin, with half-angle `tilt`."""
    u=np.cross(target,(1.0,0,0) if abs(target[0])<.9 else (0,1.0,0));u/=np.linalg.norm(u);v=np.cross(target,u)
    t=np.radians(tilt);phi=np.linspace(0,2*np.pi,count,endpoint=False)
    rim=length*(np.cos(t)*target+np.sin(t)*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v))
    circle=np.stack([rim,np.roll(rim,-1,axis=0)],axis=1)
    spokes=np.stack([np.zeros((4,3)),rim[::count//4]],axis=1)
    axis=np.array([[np.zeros(3),1.25*length*target]])
    return np.concatenate([axis,spokes,circle]).astype(np.float32)


def main():
    global WALL,PLANNER,EVAL,METHODS,COLORS
    p=argparse.ArgumentParser();p.add_argument('--evaluation-dir',type=Path,help='Replay a newly generated evaluation folder without optional frozen assets');p.add_argument('--wall',action='store_true');p.add_argument('--planner',choices=('mppi','rk4'),default='mppi');p.add_argument('--port',type=int,default=8901);p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--scene',help='Scene to draw and use for orientation rules: a YAML path or a name in experiments/transition_wall/scenes (e.g. wall40)');a=p.parse_args()
    if a.evaluation_dir is not None:EVAL=a.evaluation_dir.resolve()
    WALL=a.wall;PLANNER=a.planner
    if WALL:
        if a.evaluation_dir is None:EVAL=ROOT/'results/registered_20260930T013432Z-7905f45e/evaluation'
        METHODS=('Wall retraining '+PLANNER.upper(),'No-wall weights, direct transfer '+PLANNER.upper())
        COLORS=((55,185,255),(255,165,60))
    server=viser.ViserServer(host=a.host,port=a.port,label='Clean Bellman · 3 losses')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/grid',width=2,height=2,cell_size=.1,plane='xy')
    server.scene.add_light_ambient('/ambient',intensity=.7)
    server.scene.add_light_directional('/key',intensity=1.2,wxyz=(.85,.35,.35,.15))
    scene_file=ROOT/'experiments/wall25/scene.yml' if WALL else RELEASE/'configs/scene.yml'
    if a.scene:
        scene_file=Path(a.scene) if Path(a.scene).exists() else ROOT/'experiments/transition_wall/scenes'/(a.scene+'.yml')
        if not scene_file.exists():raise SystemExit(f'Scene not found: {a.scene}')
    helper.add_scene_yaml_to_viser(server,str(scene_file),draw_zones=True)
    urdf=helper.load_urdf(str(RELEASE/'assets/runtime/robot/ur5e_robotiq_2f_140.urdf'))
    robot=helper.add_robot_scaled(server,urdf)
    suites=json.loads((RELEASE/'tests/suite_manifest.json').read_text())['suites']
    types={};zones={}
    for item in suites:
        with np.load(RELEASE/'tests/suites'/item['file'],allow_pickle=True) as z:
            types[item['task']]=z['pair_type'].copy();zones[item['task']]=json.loads(str(z['corner_zones_json']))
    rules=gripper_rules(yaml.safe_load(scene_file.read_text())['task_axes'])
    gripper=server.scene.add_frame('/gripper_frame',axes_length=.09,axes_radius=.004,origin_radius=.008)
    cones={name:server.scene.add_line_segments('/gripper_targets/'+name,points=tolerance_cone(rule['target']),
                                               colors=AXIS_COLORS[rule['axis']],line_width=3)
           for name,rule in rules.items()}
    summary=json.loads((EVAL/'evaluation.json').read_text()) if (EVAL/'evaluation.json').exists() else {}
    sr=100*sum(summary.get('mppi_successes',[532,533,532]))/1800
    if WALL:
        (retrain_ok,retrain_n),(transfer_ok,transfer_n)=success_total(METHODS[0]),success_total(METHODS[1])
        source='' if TRANSFER.exists() else ' Transfer paths are the no-wall baseline paths; their wall verdicts come from recorded evidence.'
        header=(f'Wall retraining comparison · {PLANNER.upper()}\nBlue: wall retraining ({EVAL.parent.name}); orange: no-wall weights, direct transfer.'
                f' Both are scored in the same 25 cm wall scene. {PLANNER.upper()}: {100*retrain_ok/retrain_n:.2f}% vs {100*transfer_ok/transfer_n:.2f}%.{source}'
                '\n\nDrag to rotate, scroll to zoom; tick Play to watch the robot move. Lines show only the end-effector path; whole-arm safety is decided by the strict scorer.')
    elif a.evaluation_dir is not None or a.scene:
        (mppi_ok,mppi_n),(rk4_ok,rk4_n)=success_total(METHODS[0],(0,1,2)),success_total(METHODS[1],(0,))
        header=(f'{EVAL.parent.name} · scene {scene_file.stem}\nBlue: Bellman MPPI {100*mppi_ok/mppi_n:.2f}% ({mppi_ok}/{mppi_n}); '
                f'purple: Bellman RK4 {100*rk4_ok/rk4_n:.2f}% ({rk4_ok}/{rk4_n}). Replay of saved trajectories.')
    else:
        header='Clean baseline · 338K parameters · 88.72%\nBlue: Bellman MPPI; purple: Bellman RK4. Replay of frozen trajectories.'
    audit=summary.get('endpoint_audit',{}).get('tasks')
    server.gui.add_markdown('# '+header)
    filter_options=('All','Cross (cross-region)','Rescued by retraining','New failure after retraining','Both succeed','Both fail') if WALL else ('All','Cross (cross-region)','MPPI fails / RK4 succeeds','MPPI succeeds / RK4 fails')
    task=server.gui.add_dropdown('Task',options=tuple(types),initial_value='mixed')
    method=server.gui.add_dropdown('Robot follows',options=METHODS,initial_value=METHODS[0])
    seed=server.gui.add_dropdown('Planning seed',options=('0','1','2'),initial_value='0')
    filt=server.gui.add_dropdown('Filter',options=filter_options,initial_value='Rescued by retraining' if WALL else 'Cross (cross-region)')
    query=server.gui.add_dropdown('Query',options=('60',),initial_value='60')
    compare=server.gui.add_checkbox('Show both methods\' paths',initial_value=True)
    frame=server.gui.add_slider('Trajectory frame',min=0,max=1,step=1,initial_value=0)
    play=server.gui.add_checkbox('Play',initial_value=False)
    fps=server.gui.add_slider('Playback speed (frames/s)',min=1,max=60,step=1,initial_value=20)
    info=server.gui.add_markdown('Loading…')
    show_gripper=server.gui.add_checkbox('Show gripper frame and required directions',initial_value=True)
    def describe(rule):
        direction=WORLD_NAMES.get(tuple(int(v) for v in np.round(rule['target'])),str(rule['target']))
        return f"{AXIS_COLOR_NAMES[rule['axis']]} cone: gripper **+{rule['axis']}** must point **{direction}**, within {TILT_DEG:g}°"
    server.gui.add_markdown('### Gripper orientation\nAxes at the grasp frame: red +x, green +y, blue +z.\n\n'
        +'\n'.join(f"- **{RULE_ZONES[name]}** zone, {describe(rule)}" for name,rule in rules.items())
        +'\n- **transition** band: either rule is enough, so both cones are shown.\n\n'
        'A cone is drawn only where its rule applies; the gripper axis of the same colour must stay inside it.')
    orientation=server.gui.add_markdown('')
    lock=threading.RLock();state={'busy':False,'path':None,'line_handles':[]}

    def show_orientation(index):
        transform=helper.get_link_transform(urdf,'grasp_frame');R=transform[:3,:3];origin=transform[:3,3]
        x,y=float(origin[0]),float(origin[1]);zone=zone_of(x,y,zones[task.value])
        tilts={name:tilt_deg(R,rule) for name,rule in rules.items()}
        active={'cupping':['cup'],'pouring':['pour'],'transition':list(rules)}[zone]
        ok=any(tilts[name]<=TILT_DEG for name in active)
        visible=bool(show_gripper.value)
        gripper.visible=visible;gripper.position=tuple(origin);gripper.wxyz=tuple(trimesh.transformations.quaternion_from_matrix(transform))
        for name,cone in cones.items():
            cone.visible=visible and name in active;cone.position=tuple(origin)
        rows=[f"|{RULE_ZONES[name]}|{tilts[name]:.1f}°|{'checked' if name in active else 'not checked here'}|"
              +('✓' if tilts[name]<=TILT_DEG else '✗')+'|' for name in rules]
        orientation.content=(f'**Frame {index}** · grasp point at x={x:.3f}, y={y:.3f} m · zone **{zone}**\n\n'
            '|Rule|Tilt|In this zone|≤20°|\n|---|---:|---|:-:|\n'+'\n'.join(rows)
            +f"\n\nThis frame: **{'OK' if ok else 'VIOLATES the tilt rule'}**\n\n"
            '_Only saved frames are shown; the scorer also checks interpolated points every 0.001 rad, '
            'so a run can fail between frames that all look OK here._')

    def show():
        with lock:
            if state['path'] is not None:
                index=min(int(frame.value),len(state['path'])-1)
                helper.update_robot(urdf,robot,state['path'][index])
                show_orientation(index)

    def refresh_choices():
        b=load(task.value,METHODS[0],int(seed.value))[1];r=load(task.value,METHODS[1],int(seed.value))[1]
        ids=list(range(100))
        if filt.value=='Cross (cross-region)':ids=[i for i in ids if types[task.value][i]==2]
        elif filt.value in ('MPPI fails / RK4 succeeds','New failure after retraining'):ids=[i for i in ids if not b[str(i)]['success'] and r[str(i)]['success']]
        elif filt.value in ('MPPI succeeds / RK4 fails','Rescued by retraining'):ids=[i for i in ids if b[str(i)]['success'] and not r[str(i)]['success']]
        if filt.value=='Both succeed':ids=[i for i in ids if b[str(i)]['success'] and r[str(i)]['success']]
        elif filt.value=='Both fail':ids=[i for i in ids if not b[str(i)]['success'] and not r[str(i)]['success']]
        options=tuple(map(str,ids)) or ('No matching queries',)
        old=query.value;query.options=options;query.value=old if old in options else options[0]

    def redraw(reset_choices=False):
        with lock:
            if state['busy']:return
            state['busy']=True
            try:
                play.value=False
                if reset_choices:refresh_choices()
                for h in state['line_handles']:h.remove()
                state['line_handles']=[]
                if not query.value.isdigit():
                    state['path']=None;info.content='No queries match the current filter.';return
                qi=int(query.value);details=[]
                for index,name in enumerate(METHODS):
                    paths,scores=load(task.value,name,int(seed.value));path=paths[qi];score=scores[str(qi)]
                    details.append(f'**{name}**: '+('success' if score['success'] else 'failure — '+', '.join(score['reasons'])))
                    if compare.value or name==method.value:
                        ee=helper.ee_path(urdf,path,maximum=len(path))
                        state['line_handles'].append(server.scene.add_line_segments('/paths/'+str(index),points=helper.segments(ee),colors=COLORS[index],line_width=4 if name==method.value else 2))
                    if name==method.value:state['path']=path
                path=state['path'];frame.value=0;frame.max=max(1,len(path)-1)
                typ=('Cupping','Pouring','Cross')[int(types[task.value][qi])]
                if audit and not audit[task.value]['per_query'][qi]['wall_endpoints_valid']:
                    details.append('**Endpoints blocked by the wall**: this query is impossible in this scene.')
                info.content=f'### {task.value} / #{qi} / {typ}\n\n'+'\n\n'.join(details)+f'\n\nCurrent trajectory: {len(path)} saved frames.\n\nSuccess/failure comes from the original dense scoring; replay does not modify trajectories and draws no extra smoothed curves.'
                show()
            finally:state['busy']=False

    for handle in (task,seed,filt):
        handle.on_update(lambda _:redraw(True))
    for handle in (query,method,compare):
        handle.on_update(lambda _:redraw(False))
    frame.on_update(lambda _:show())
    show_gripper.on_update(lambda _:show())
    @server.on_client_connect
    def connected(client):
        client.camera.position=(1.6,-1.6,1.25);client.camera.look_at=(.5,0,.25);client.camera.up=(0,0,1)
    redraw(True)
    print(f'READY http://localhost:{a.port}',flush=True)
    while True:
        time.sleep(1/float(fps.value))
        if play.value and state['path'] is not None:
            with lock:
                frame.value=(int(frame.value)+1)%len(state['path']);show()


if __name__=='__main__':main()
