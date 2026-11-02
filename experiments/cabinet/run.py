"""Register cabinet-task work (experiments/cabinet/scenes/<scene>.yml, --scene, default cabinet).

cabinet-tests   generate the certified test suites of a scene (frozen afterwards by freeze_tests.py)
cabinet         Bellman from source: FK pool and clearance labels, speed, bank, TravelCost, frozen suites
cabinet-metric  metric NTFields baseline (ntrl-demo): cabinet data, ntrl-demo training, frozen suites
cabinet-evaluate  evaluate an existing cabinet or cabinet-metric run's final checkpoint on the frozen suites

Scenes: cabinet, and cabinet_plus10 / cabinet_plus20 with 10% / 20% more clearance (make_scenes.py).
Frozen suites: tests/ for cabinet, tests_plus10/ and tests_plus20/ for the others.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
import yaml
from launch import HERE,RELEASE,IMAGE
from tools.runtime_assets import verify

EXPERIMENT=HERE/'experiments/cabinet'


def scene_paths(scene):
 """Scene file and frozen test suites of a cabinet scene."""
 if not scene.startswith('cabinet') or not (EXPERIMENT/'scenes'/(scene+'.yml')).is_file():raise SystemExit(f'No cabinet scene {scene} in {EXPERIMENT/"scenes"}')
 return EXPERIMENT/'scenes'/(scene+'.yml'),EXPERIMENT/('tests'+scene[len('cabinet'):])


def execute(a):
 verify()
 registry=HERE/'tools/experiment_registry/registry.py'
 gpus='all' if a.gpu is None else f'device={a.gpu}'
 scene=a.scene or 'cabinet';SCENE,TESTS=scene_paths(scene);doc=yaml.safe_load(SCENE.read_text())
 scene_sha=hashlib.sha256(SCENE.read_bytes()).hexdigest();variant=[] if scene=='cabinet' else [scene]
 # Single files rather than the experiment folder, so work on one method or the report while
 # another run is in progress does not invalidate that run's input identity.
 inputs=dict(scene=str(SCENE),registration=str(EXPERIMENT/'run.py'),geometry=str(EXPERIMENT/'geometry.py'),method=str(HERE/'method'),runtime=str(RELEASE),
  launcher=str(HERE/'launch.py'),entrypoint=str(HERE/'run.py'),runtime_manifest=str(HERE/'runtime_manifest.json'),runtime_tools=str(HERE/'tools/runtime_assets.py'),
  registry=str(registry))
 common=dict(action=a.action,scene=scene,scene_sha256=scene_sha,smoke=a.smoke,gpus=gpus,orientation_rule=None,
  sampling_box=dict(**{k:doc['sampling'][k] for k in 'xyz'},rule=doc['sampling'].get('rule','height = cupboard, width = 1.5 x, depth = 2.5 x from the back face')))
 if variant:common.update(clearance_scale=doc['variant']['clearance_scale'],clearance_rule=doc['variant']['rule'])
 if a.action=='cabinet-tests':
  name='Clean Bellman600 cabinet-tests'+''.join(' '+v for v in variant)
  config=dict(common,per_stratum=3 if a.smoke else 100,seeds=dict(pool=20261014,pair=20261015,rrt=20261016,select=20261017),
   witness_planners=['direct line','RRT-Connect'],certificate='release scorer, tilt gate off')
  inputs.update(generator=str(EXPERIMENT/'make_tests.py'),obstacle_generator=str(HERE/'experiments/obstacle_scenes/make_tests.py'))
  required=['tests/suite_manifest.json','tests/generation.json','tests/suites/*.npz']
 else:
  if not (TESTS/'suite_manifest.json').exists():raise SystemExit(f'Freeze the {scene} test suites into {TESTS} first (run.py cabinet-tests, freeze_tests.py).')
  inputs.update(pipeline=str(EXPERIMENT/'pipeline.py'),evaluate=str(EXPERIMENT/'evaluate.py'),shared_evaluate=str(HERE/'experiments/transition_wall/evaluate.py'),
   scene_tests=str(TESTS),data_generation=str(HERE/'data_generation'))
  common.update(planning_seeds=[0] if a.smoke else [0,1,2],scene_tests=str(TESTS.relative_to(HERE)),scorer_tilt_deg=180)
  if a.action=='cabinet-evaluate':
   if not a.run:raise SystemExit('cabinet-evaluate needs --run results/registered_<id>')
   run=HERE/'results'/Path(a.run).name;status=json.loads((run/'status.json').read_text())
   progress=json.loads((run/'training/progress.json').read_text());checkpoint=run/'training/checkpoint.pt'
   if status['smoke']!=a.smoke or not progress.get('complete') or not checkpoint.exists():raise SystemExit(f'{run.name} has no finished final checkpoint')
   if Path(status['scene']).stem!=scene:raise SystemExit(f'{run.name} was trained in {Path(status["scene"]).stem}, not {scene}')
   method=status['method'];name=f'Clean Bellman600 cabinet-evaluate {method} {run.name}'
   config=dict(common,method=method,run=run.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
   if method=='metric':config.update(ntrl_mppi_seeds=[0] if a.smoke else [0,1,2])
   inputs.update(checkpoint=str(checkpoint),run_status=str(run/'status.json'),run_progress=str(run/'training/progress.json'))
   if method=='metric':inputs.update(metric_field=str(EXPERIMENT/'metric/field.py'),ntrl_network=str(EXPERIMENT/'metric/ntrl_demo/models/metric/model_network_metric.py'))
   required=['evaluation/evaluation.json','evaluation/REPORT.md']
  elif a.action=='cabinet':
   name='Clean Bellman600 cabinet bellman from-source'+''.join(' '+v for v in variant)
   config=dict(common,method='bellman',fk_seed=42,pair_seed=42,coverage_seed=20260912,speed_seed=87358,bank_seed=87344,training_seed=87358,
    fk_pool=4096 if a.smoke else 2000000,speed_steps=12 if a.smoke else 20000,bank_walkers=512 if a.smoke else 50000,bank_generations=2 if a.smoke else 8,
    losses=dict(bellman=6,eikonal=2,direction=2),rounds=2 if a.smoke else 5,speed_label='clearance only',precomputed_training_inputs=False)
   inputs.update(generate=str(EXPERIMENT/'generate.py'),prepare=str(EXPERIMENT/'prepare.py'),training=str(HERE/'experiments/from_source/training.py'),
    bank=str(HERE/'experiments/from_source/bank.py'))
   required=['status.json','sampled/complete.json','sampled/inputs/source_manifest.json','prepared/complete.json','prepared/speed.pt',
    'prepared/bank/manifest.json','training/checkpoint.pt','evaluation/evaluation.json','evaluation/REPORT.md']
  else:
   name='Clean Bellman600 cabinet metric-ntfields baseline'+''.join(' '+v for v in variant)
   config=dict(common,method='metric_ntfields',source='ntrl-demo 5ed065f6040bac0003fc0cf69874e1bdfbc50a6a (vendored, unmodified)',data_seed=20261019,
    pairs=4096 if a.smoke else 500000,epochs=2 if a.smoke else 5000,training_seed=87358,speed_label='clearance only',ntrl_mppi_seeds=[0] if a.smoke else [0,1,2])
   inputs.update(metric_data=str(EXPERIMENT/'metric/data.py'),metric_train=str(EXPERIMENT/'metric/train.py'),metric_field=str(EXPERIMENT/'metric/field.py'),
    ntrl_demo=str(EXPERIMENT/'metric/ntrl_demo'))
   required=['status.json','data/complete.json','training/checkpoint.pt','training/progress.json','evaluation/evaluation.json','evaluation/REPORT.md']
 spec=dict(name=name+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),config=config,
  environment=dict(docker_context='default',docker_image_id=IMAGE),inputs=inputs,
  command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}','--gpus',gpus]+[x for v in variant for x in ['--scene',v]]+(['--run',run.name] if a.action=='cabinet-evaluate' else [])+(['--smoke'] if a.smoke else []),
  required_outputs=required,validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True)
 path=directory/('_'.join([a.action]+variant+([run.name] if a.action=='cabinet-evaluate' else []))+('_smoke' if a.smoke else '')+'.json');path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 subprocess.run([sys.executable,str(registry),'list','--search',a.action],check=True)
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
