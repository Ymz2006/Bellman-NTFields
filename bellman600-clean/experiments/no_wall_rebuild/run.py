"""Register the full scene-specific offline training pipeline."""
import json,subprocess,sys
from pathlib import Path
from launch import HERE,RELEASE,IMAGE

def execute(a):
 registry=HERE/'tools/experiment_registry/registry.py'
 spec=dict(name='Clean Bellman600 no-wall input rebuild'+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),
  config=dict(action='rebuild-reproduce',smoke=a.smoke,scene='no_wall_overlap010',speed_seed=87358,bank_seed=87344,training_seed=87358,
   speed_steps=12 if a.smoke else 20000,walkers=512 if a.smoke else 50000,generations=2 if a.smoke else 8,
   training_rounds=2 if a.smoke else 5,epochs_per_round=1 if a.smoke else 2,
   training_updates='4 smoke; otherwise10*ceil(retained_bank_sources/256)',loss_weights=dict(bellman=6,eikonal=2,direction=2),
   planning_seeds=[0] if a.smoke else [0,1,2]),
  environment=dict(docker_context='default',docker_image_id=IMAGE),
  inputs=dict(experiment=str(HERE/'experiments/no_wall_rebuild'),scene=str(RELEASE/'configs/scene.yml'),
   method=str(HERE/'method'),launcher=str(HERE/'launch.py'),
   frozen_speed=str(HERE/'assets/speed.pt'),frozen_bank=str(HERE/'assets/bank'),
   training_sources=str(HERE/'training_inputs/no_wall_seed_pool'),reference=str(HERE/'assets/reference'),runtime=str(RELEASE),entrypoint=str(HERE/'run.py')),
  command=[sys.executable,str(HERE/'launch.py'),'rebuild-reproduce','--output','{output}']+(['--smoke'] if a.smoke else []),
  required_outputs=['status.json','prepared/complete.json','prepared/speed.pt','prepared/bank/manifest.json','training/checkpoint.pt','evaluation/evaluation.json'],validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True);path=directory/('rebuild-reproduce'+('_smoke' if a.smoke else '')+'.json')
 path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 subprocess.run([sys.executable,str(registry),'list','--search','Bellman600 no-wall'],check=True)
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True)
 (directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not (check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Registry check blocked; inspect '+str(path))
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
