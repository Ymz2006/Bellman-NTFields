"""Register a speed-network test for one existing run (read-only: no training, no planning).

speed-test  compare the run's prepared/speed.pt with ground-truth physical speed on sampled configurations
"""
import hashlib,json,subprocess,sys
from pathlib import Path
from launch import HERE,RELEASE,IMAGE
from tools.runtime_assets import verify

EXPERIMENT=HERE/'experiments/speed_test'

def execute(a):
 if not a.run:raise SystemExit('speed-test needs --run results/registered_<id>')
 run=HERE/'results'/Path(a.run).name;speed=run/'prepared/speed.pt'
 if not speed.exists():raise SystemExit(f'{run.name} has no prepared/speed.pt')
 status=json.loads((run/'status.json').read_text())
 scene=HERE/Path(status['scene']).relative_to('/clean') if status.get('scene') else RELEASE/'configs/scene.yml'
 verify()
 registry=HERE/'tools/experiment_registry/registry.py'
 gpus='all' if a.gpu is None else f'device={a.gpu}'
 inputs=dict(speed=str(speed),scene=str(scene),test=str(EXPERIMENT/'test_speed.py'),registration=str(EXPERIMENT/'run.py'),
  physical=str(HERE/'experiments/from_source/physical.py'),geometry=str(HERE/'experiments/from_source/geometry.py'),method=str(HERE/'method'),
  runtime=str(RELEASE),launcher=str(HERE/'launch.py'),entrypoint=str(HERE/'run.py'),runtime_manifest=str(HERE/'runtime_manifest.json'),registry=str(registry))
 if (run/'prepared/speed_audit.json').exists():inputs['speed_audit']=str(run/'prepared/speed_audit.json')
 if (run/'sampled/inputs').exists():inputs['training_inputs']=str(run/'sampled/inputs')
 config=dict(action=a.action,run=run.name,speed_sha256=hashlib.sha256(speed.read_bytes()).hexdigest(),scene=scene.name,
  scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),samples=512 if a.smoke else 20000,sample_seed=20261018,
  sets=['uniform','uniform_valid','heldout_original','heldout_augmentation'] if (run/'sampled/inputs').exists() else ['uniform','uniform_valid'],smoke=a.smoke,gpus=gpus)
 spec=dict(name=f'Clean Bellman600 speed-test {run.name}'+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),config=config,
  environment=dict(docker_context='default',docker_image_id=IMAGE),inputs=inputs,
  command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}','--run',run.name,'--gpus',gpus]+(['--smoke'] if a.smoke else []),
  required_outputs=['speed_test/speed_test.json','speed_test/REPORT.md','speed_test/samples.npz'],validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True)
 path=directory/(f'{a.action}_{run.name}'+('_smoke' if a.smoke else '')+'.json');path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 subprocess.run([sys.executable,str(registry),'list','--search',a.action],check=True)
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
