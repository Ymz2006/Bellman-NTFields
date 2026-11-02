"""Portable registered entry point for the frozen offline Bellman baseline."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from launch import HERE,RELEASE,IMAGE
from tools.assets import verify

REGISTRY=HERE/'tools/experiment_registry/registry.py'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['doctor','train','evaluate','reproduce','wall-audit','wall-evaluate','wall-reproduce','rebuild-reproduce','from-source','from-source-prepare','doctor-source','transition-wall','obstacle-scene','obstacle-tests','obstacle-evaluate','speed-test','cabinet-tests','cabinet','cabinet-metric','cabinet-evaluate','face-speed','face-speed-evaluate','face-speed-accuracy','face-speed-guard','point-field','point-field-bellman','ntrl-cabinet','pntfields-cabinet'])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--self-net',help='point-field: split run whose self-collision network replaces the exact self term in the learned-self controls')
    p.add_argument('--speed',choices=['learned','exact'],help='point-field-bellman: hierarchical speed with the learned point field (--run: the point-field run) or the exact scene distance')
    p.add_argument('--reference',help='point-field-bellman: baseline rerun whose seed states and bank transitions must be reproduced')
    p.add_argument('--repeat-reason')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--scene',help='transition-wall: scene in experiments/transition_wall/scenes (wall40 or wall80); obstacle-scene: scene in experiments/obstacle_scenes/scenes (platforms, shelf or clutter); cabinet*: scene in experiments/cabinet/scenes (cabinet, cabinet_plus10 or cabinet_plus20; default cabinet)')
    p.add_argument('--gpu',help='transition-wall, obstacle-*, speed-test, cabinet*: run on this GPU index instead of all GPUs')
    p.add_argument('--run',help='obstacle-evaluate, speed-test, cabinet-evaluate: registered run (results/registered_<id>) whose checkpoint to evaluate or speed network to test')
    p.add_argument('--ntrl-args',help='ntrl-cabinet: options passed verbatim to experiments/ntrl_cabinet/pipeline.py, e.g. "--variant metric_arm --coords ntrl"')
    p.add_argument('--pnt-args',help='pntfields-cabinet: options passed verbatim to experiments/pntfields_cabinet/pipeline.py, e.g. "--joint-limit 3.14159"')
    p.add_argument('--suite',choices=['scene','frozen600'],default='scene',help="obstacle-evaluate: the scene's frozen test suites (default) or the frozen 600 queries")
    p.add_argument('--method',choices=['bellman','metric'],help='face-speed: which field to train with the per-piece speed')
    p.add_argument('--cover',choices=['faces','spheres','links','split','environment','release'],help='face-speed: pieces of the speed (faces of the box cover, spheres, links, environment+self split, environment only without self-collision, or the release label)')
    p.add_argument('--speed-model',choices=['shared','separate'],default='shared',help='face-speed bellman: one trunk with an output per piece (default) or one separate network per piece')
    p.add_argument('--data',help='face-speed metric: registered run whose data/ to reuse (baseline rerun)')
    p.add_argument('--sampled',help='face-speed bellman: registered cabinet run whose sampled/ FK pool and coverage set to reuse')
    p.add_argument('--speed-steps',type=int,default=20000,help='face-speed bellman: speed network steps (default 20000)')
    p.add_argument('--runs',nargs='*',help='face-speed-accuracy: registered runs whose prepared/speed.pt to test')
    p.add_argument('--filter',choices=['cover','release'],default='cover',help='face-speed metric: clearance used by the x0 near-obstacle filter (default: the cover clearance)')
    p.add_argument('--margin',type=float,help='face-speed metric: speed label margin in m (default 0.05)')
    p.add_argument('--offset',type=float,help='face-speed metric: speed label offset (lower clip) in m (default 0.005)')
    p.add_argument('--pairs',type=int,help='face-speed metric: training pairs (default 500000)')
    p.add_argument('--rows',choices=['pieces','body','none'],default='pieces',help='face-speed-guard: one constraint per piece (default), one for the body speed, or none (control)')
    p.add_argument('--alpha',type=float,default=.5,help='face-speed-guard: barrier rate (default 0.5)')
    p.add_argument('--floor',type=float,default=.15,help='face-speed-guard: speed floor (default 0.15)')
    a=p.parse_args()
    if a.action=='transition-wall':
        from experiments.transition_wall.run import execute
        execute(a)
        return
    if a.action in ('obstacle-scene','obstacle-tests','obstacle-evaluate'):
        from experiments.obstacle_scenes.run import execute
        execute(a)
        return
    if a.action=='speed-test':
        from experiments.speed_test.run import execute
        execute(a)
        return
    if a.action=='ntrl-cabinet':
        from experiments.ntrl_cabinet.run import execute
        execute(a)
        return
    if a.action=='pntfields-cabinet':
        from experiments.pntfields_cabinet.run import execute
        execute(a)
        return
    if a.action in ('point-field','point-field-bellman'):
        from experiments.point_field.run import execute
        execute(a)
        return
    if a.action.startswith('face-speed'):
        from experiments.face_speed.run import execute
        execute(a)
        return
    if a.action in ('cabinet-tests','cabinet','cabinet-metric','cabinet-evaluate'):
        from experiments.cabinet.run import execute
        execute(a)
        return
    if a.action in ('from-source','from-source-prepare','doctor-source'):
        from experiments.from_source.run import execute
        execute(a)
        return
    verify()
    if a.action=='rebuild-reproduce':
        from experiments.no_wall_rebuild.run import execute
        execute(a)
        return
    if a.action=='wall-reproduce':
        from experiments.wall25_train.run import execute
        execute(a)
        return
    if a.action.startswith('wall-'):
        from experiments.wall25.run import execute
        execute(a)
        return
    if a.action=='doctor':
        subprocess.run(['docker','--context','default','image','inspect',IMAGE,'--format','{{.Id}}'],check=True)
        print('Assets verified; pinned Docker image found. GPU execution is checked by reproduce --smoke.')
        return
    required={'train':['training/checkpoint.pt','training/progress.json'],
        'evaluate':['evaluation/evaluation.json','evaluation/REPORT.md'],
        'reproduce':['status.json','training/checkpoint.pt','evaluation/evaluation.json']}[a.action]
    spec=dict(name='Standalone Clean Bellman600 '+a.action+(' smoke' if a.smoke else ' full'),
        kind='pipeline',root=str(HERE),config=dict(action=a.action,smoke=a.smoke,
            training_seed=87358,planning_seeds=[0] if a.smoke else [0,1,2],
            training_steps=4 if a.smoke else 15620,loss_weights={'bellman':6,'eikonal':2,'direction':2}),
        environment=dict(docker_context='default',docker_image_id=IMAGE),
        inputs=dict(method=str(HERE/'method'),launcher=str(HERE/'launch.py'),
            pipeline=str(HERE/'tools/reproduce.py'),bank=str(HERE/'assets/bank'),
            speed=str(HERE/'assets/speed.pt'),checkpoint=str(HERE/'assets/checkpoint.pt'),
            reference=str(HERE/'assets/reference'),runtime=str(RELEASE),
            entrypoint=str(HERE/'run.py'),registry=str(REGISTRY)),
        command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}']+(['--smoke'] if a.smoke else []),
        required_outputs=required,validation='files')
    directory=HERE/'specs';directory.mkdir(exist_ok=True)
    path=directory/(a.action+('_smoke' if a.smoke else '')+'.json')
    path.write_text(json.dumps(spec,indent=2)+'\n')
    if a.prepare_only:
        print(path);return
    subprocess.run([sys.executable,str(REGISTRY),'list','--search','Clean Bellman600'],check=True)
    check=subprocess.run([sys.executable,str(REGISTRY),'check',str(path)],capture_output=True,text=True)
    check_path=directory/(path.stem+'.check.json');check_path.write_text(check.stdout)
    if check.returncode and not (check.returncode==3 and a.repeat_reason):
        raise SystemExit(check.stderr or f'Registry blocked execution. Inspect {check_path}; exact repeats require --repeat-reason.')
    command=[sys.executable,str(REGISTRY),'run',str(path)]
    if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
    subprocess.run(command,check=True)

if __name__=='__main__':main()
