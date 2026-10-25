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
    p.add_argument('action',choices=['doctor','train','evaluate','reproduce','wall-audit','wall-evaluate','wall-reproduce','rebuild-reproduce','from-source','from-source-prepare','doctor-source','transition-wall','obstacle-scene','obstacle-tests','obstacle-evaluate','speed-test','cabinet-tests','cabinet','cabinet-metric','cabinet-evaluate'])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--repeat-reason')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--scene',help='transition-wall: scene in experiments/transition_wall/scenes (wall40 or wall80); obstacle-scene: scene in experiments/obstacle_scenes/scenes (platforms, shelf or clutter); cabinet*: scene in experiments/cabinet/scenes (cabinet, cabinet_plus10 or cabinet_plus20; default cabinet)')
    p.add_argument('--gpu',help='transition-wall, obstacle-*, speed-test, cabinet*: run on this GPU index instead of all GPUs')
    p.add_argument('--run',help='obstacle-evaluate, speed-test, cabinet-evaluate: registered run (results/registered_<id>) whose checkpoint to evaluate or speed network to test')
    p.add_argument('--suite',choices=['scene','frozen600'],default='scene',help="obstacle-evaluate: the scene's frozen test suites (default) or the frozen 600 queries")
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
