"""Register obstacle-scene work for one scene (platforms, shelf or clutter).

obstacle-scene     from-source pipeline: training, frozen-600 evaluation and the scene's frozen test suites
obstacle-tests     generate the scene's certified test suites (frozen afterwards into tests/<scene>/)
obstacle-evaluate  evaluate an existing obstacle-scene run's checkpoint on the scene's frozen test suites
"""
import hashlib,json,subprocess,sys
from pathlib import Path
import yaml
from launch import HERE,RELEASE,IMAGE
from tools.runtime_assets import verify

EXPERIMENT=HERE/'experiments/obstacle_scenes'
SCENES=EXPERIMENT/'scenes'
STAGES=HERE/'experiments/transition_wall'

def execute(a):
 if not a.scene:raise SystemExit(f'{a.action} needs --scene ('+', '.join(sorted(p.stem for p in SCENES.glob('*.yml')))+')')
 scene=SCENES/(a.scene+'.yml')
 if not scene.exists():raise SystemExit(f'Unknown scene {a.scene!r}; expected a file in {SCENES}')
 verify()
 registry=HERE/'tools/experiment_registry/registry.py'
 tests=EXPERIMENT/'tests'/a.scene
 gpus='all' if a.gpu is None else f'device={a.gpu}'
 scene_sha=hashlib.sha256(scene.read_bytes()).hexdigest()
 # Single files rather than the experiment folder, so editing another scene, the targets or the
 # preview while a scene runs does not invalidate that run's input identity.
 inputs=dict(scene=str(scene),registration=str(EXPERIMENT/'run.py'),method=str(HERE/'method'),runtime=str(RELEASE),launcher=str(HERE/'launch.py'),
  entrypoint=str(HERE/'run.py'),runtime_manifest=str(HERE/'runtime_manifest.json'),runtime_tools=str(HERE/'tools/runtime_assets.py'),registry=str(registry))
 extra=[]
 if a.action=='obstacle-scene':
  if not (tests/'suite_manifest.json').exists():raise SystemExit(f'Freeze the {a.scene} test suites into {tests} first (run.py obstacle-tests).')
  obstacles=[{k:c[k] for k in ('name','pose','dims')} for c in yaml.safe_load(scene.read_text())['cuboids'][1:]]
  name=f'Clean Bellman600 obstacle-scene {a.scene} from-source'
  config=dict(action=a.action,scene=a.scene,scene_sha256=scene_sha,obstacles=obstacles,smoke=a.smoke,gpus=gpus,
   fk_seed=42,pair_seed=42,coverage_seed=20260912,speed_seed=87358,bank_seed=87344,training_seed=87358,
   fk_pool_each=4096 if a.smoke else 2000000,speed_steps=12 if a.smoke else 20000,bank_walkers=512 if a.smoke else 50000,
   bank_generations=2 if a.smoke else 8,losses=dict(bellman=6,eikonal=2,direction=2),rounds=2 if a.smoke else 5,
   precomputed_training_inputs=False,planning_seeds=[0] if a.smoke else [0,1,2],scene_tests=str(tests.relative_to(HERE)))
  inputs.update(pipeline=str(EXPERIMENT/'pipeline.py'),audit=str(STAGES/'audit.py'),evaluate=str(STAGES/'evaluate.py'),
   scene_evaluate=str(EXPERIMENT/'evaluate_tests.py'),scene_tests=str(tests),data_generation=str(HERE/'data_generation'),
   from_source=str(HERE/'experiments/from_source'),reference=str(HERE/'assets/reference'))
  required=['status.json','audit/endpoint_audit.json','sampled/complete.json','sampled/inputs/source_manifest.json','prepared/complete.json',
   'prepared/speed.pt','prepared/bank/manifest.json','training/checkpoint.pt','evaluation/evaluation.json','scene_tests/evaluation.json']
 elif a.action=='obstacle-tests':
  name=f'Clean Bellman600 obstacle-tests {a.scene}'
  config=dict(action=a.action,scene=a.scene,scene_sha256=scene_sha,smoke=a.smoke,gpus=gpus,per_stratum=3 if a.smoke else 100,
   seeds=dict(pool=20261014,pair=20261015,rrt=20261016,select=20261017),witness_planners=['direct line','RRT-Connect'],certificate='release scorer')
  inputs.update(generator=str(EXPERIMENT/'make_tests.py'),targets=str(EXPERIMENT/'targets.yml'))
  required=['tests/suite_manifest.json','tests/generation.json','tests/suites/*.npz']
 else:
  if not a.run:raise SystemExit('obstacle-evaluate needs --run results/registered_<id>')
  run=HERE/'results'/Path(a.run).name;status=json.loads((run/'status.json').read_text())
  if status.get('smoke') or Path(status['scene']).stem!=a.scene:raise SystemExit(f'{run.name} is not a full {a.scene} run')
  # A pipeline interrupted after training still has its final round-5 checkpoint (written last).
  progress=run/'training/progress.json'
  trained=progress.exists() and json.loads(progress.read_text()).get('complete') and (run/'training/checkpoint.pt').exists()
  if not (status.get('complete') or trained):raise SystemExit(f'{run.name} has no finished round-5 checkpoint')
  checkpoint=run/'training/checkpoint.pt'
  suite=getattr(a,'suite',None) or 'scene'
  name=f'Clean Bellman600 obstacle-evaluate {a.scene} {run.name}'+(' frozen600' if suite=='frozen600' else '')
  config=dict(action=a.action,scene=a.scene,scene_sha256=scene_sha,run=run.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
   smoke=a.smoke,gpus=gpus,planning_seeds=[0] if a.smoke else [0,1,2])
  inputs.update(checkpoint=str(checkpoint),evaluate=str(STAGES/'evaluate.py'))
  extra=['--run',run.name]
  if suite=='frozen600':
   audit=run/'audit/endpoint_audit.json'
   config.update(suite='frozen600',endpoint_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
   inputs.update(endpoint_audit=str(audit))
   extra+=['--suite','frozen600']
  else:
   if not (tests/'suite_manifest.json').exists():raise SystemExit(f'Freeze the {a.scene} test suites into {tests} first (run.py obstacle-tests).')
   config.update(scene_tests=str(tests.relative_to(HERE)))
   inputs.update(scene_tests=str(tests),scene_evaluate=str(EXPERIMENT/'evaluate_tests.py'))
  required=['evaluation/evaluation.json','evaluation/REPORT.md']
 spec=dict(name=name+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),config=config,
  environment=dict(docker_context='default',docker_image_id=IMAGE),inputs=inputs,
  command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}','--scene',a.scene,'--gpus',gpus]+extra+(['--smoke'] if a.smoke else []),
  required_outputs=required,validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True)
 path=directory/('_'.join([a.action,a.scene]+[x for x in extra[1:] if not x.startswith('--')])+('_smoke' if a.smoke else '')+'.json');path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 subprocess.run([sys.executable,str(registry),'list','--search',a.action],check=True)
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
