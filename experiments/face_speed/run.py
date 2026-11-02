"""Register per-piece (box-face) speed experiments on the cabinet task (experiments/face_speed/README.md).

face-speed           --method metric  --cover faces|spheres|release [--data registered_<id>/data]
                     --method bellman --cover faces|spheres|links|release --sampled registered_<id>/sampled
face-speed-evaluate  --run registered_<id>: score a finished face-speed run's checkpoint on the frozen suites
face-speed-accuracy  --runs registered_<id> ...: speed networks against the release and cover ground truth

Scenes as for cabinet* (--scene, default cabinet). The host backend (BELLMAN600_BACKEND=host) is recorded
in the spec's environment so host and pinned-image runs never share a fingerprint.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import yaml
from launch import HERE,RELEASE,IMAGE,HOST_PYTHON
from tools.runtime_assets import verify
from experiments.cabinet.run import scene_paths

EXPERIMENT=HERE/'experiments/face_speed';CABINET=HERE/'experiments/cabinet'
FILES=['robot.py','faces.py','clearance.py','metric_data.py','pipeline.py','speed_net.py','prepare_pieces.py','training_pieces.py','speed_accuracy.py','guard.py','guard_evaluate.py','run.py']


def environment():
 if os.environ.get('BELLMAN600_BACKEND','docker')=='host':return dict(backend='host-bwrap',python=HOST_PYTHON,pinned_image_id=IMAGE,same_image_as_pinned=False)
 return dict(docker_context='default',docker_image_id=IMAGE)


def execute(a):
 verify()
 registry=HERE/'tools/experiment_registry/registry.py'
 gpus='all' if a.gpu is None else f'device={a.gpu}'
 scene=a.scene or 'cabinet';SCENE,TESTS=scene_paths(scene);doc=yaml.safe_load(SCENE.read_text())
 if not (TESTS/'suite_manifest.json').exists():raise SystemExit(f'No frozen suites in {TESTS}')
 variant=[] if scene=='cabinet' else [scene]
 inputs=dict(scene=str(SCENE),scene_tests=str(TESTS),method=str(HERE/'method'),runtime=str(RELEASE),launcher=str(HERE/'launch.py'),entrypoint=str(HERE/'run.py'),
  runtime_manifest=str(HERE/'runtime_manifest.json'),runtime_tools=str(HERE/'tools/runtime_assets.py'),registry=str(registry),
  cabinet_geometry=str(CABINET/'geometry.py'),cabinet_evaluate=str(CABINET/'evaluate.py'),shared_evaluate=str(HERE/'experiments/transition_wall/evaluate.py'),
  cabinet_registration=str(CABINET/'run.py'),**{'face_speed_'+f:str(EXPERIMENT/f) for f in FILES if (EXPERIMENT/f).exists()})
 common=dict(action=a.action,scene=scene,scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(),smoke=a.smoke,gpus=gpus,orientation_rule=None,
  planning_seeds=[0] if a.smoke else [0,1,2],scene_tests=str(TESTS.relative_to(HERE)),scorer_tilt_deg=180)
 extra=[]
 if a.action=='face-speed-guard':
  if not a.run:raise SystemExit('face-speed-guard needs --run results/registered_<id> (a finished face-speed or cabinet pipeline run)')
  run=HERE/'results'/Path(a.run).name;status=json.loads((run/'status.json').read_text());progress=json.loads((run/'training/progress.json').read_text())
  checkpoint=run/'training/checkpoint.pt'
  if status['smoke']!=a.smoke or not progress.get('complete') or not checkpoint.exists():raise SystemExit(f'{run.name} has no finished final checkpoint')
  if Path(status['scene']).stem!=scene:raise SystemExit(f'{run.name} was trained in {Path(status["scene"]).stem}, not {scene}')
  method='bellman' if status['method']=='bellman' else 'metric';cover=a.cover or 'faces'
  name=f'Clean Bellman600 face-speed-guard {method} {status.get("cover","release")} {run.name} {cover} {a.rows} a{a.alpha} f{a.floor}'
  config=dict(common,method=method,field_cover=status.get('cover','release'),run=run.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
   guard=dict(cover=cover,rows=a.rows,alpha=a.alpha,floor=a.floor,sweeps=30),planners=['guarded_rk4']+(['guarded_ntrl_mppi'] if method=='metric' else []))
  inputs.update(checkpoint=str(checkpoint),run_status=str(run/'status.json'),run_progress=str(run/'training/progress.json'))
  if method=='metric':inputs.update(metric_field=str(CABINET/'metric/field.py'),ntrl_network=str(CABINET/'metric/ntrl_demo/models/metric/model_network_metric.py'))
  required=['evaluation/evaluation.json','evaluation/REPORT.md'];extra=['--run',run.name,'--cover',cover,'--rows',a.rows,'--alpha',str(a.alpha),'--floor',str(a.floor)]
 elif a.action=='face-speed-evaluate':
  if not a.run:raise SystemExit('face-speed-evaluate needs --run results/registered_<id>')
  run=HERE/'results'/Path(a.run).name;status=json.loads((run/'status.json').read_text());progress=json.loads((run/'training/progress.json').read_text())
  checkpoint=run/'training/checkpoint.pt'
  if status['smoke']!=a.smoke or not progress.get('complete') or not checkpoint.exists():raise SystemExit(f'{run.name} has no finished final checkpoint')
  if Path(status['scene']).stem!=scene:raise SystemExit(f'{run.name} was trained in {Path(status["scene"]).stem}, not {scene}')
  method=status['method'];name=f'Clean Bellman600 face-speed-evaluate {method} {status.get("cover")} {run.name}'
  config=dict(common,method=method,cover=status.get('cover'),run=run.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
  if method=='metric':config.update(ntrl_mppi_seeds=[0] if a.smoke else [0,1,2]);inputs.update(metric_field=str(CABINET/'metric/field.py'),ntrl_network=str(CABINET/'metric/ntrl_demo/models/metric/model_network_metric.py'))
  inputs.update(checkpoint=str(checkpoint),run_status=str(run/'status.json'),run_progress=str(run/'training/progress.json'))
  required=['evaluation/evaluation.json','evaluation/REPORT.md'];extra=['--run',run.name]
 elif a.action=='face-speed-accuracy':
  runs=[HERE/'results'/Path(r).name for r in (a.runs or [])]
  if not runs:raise SystemExit('face-speed-accuracy needs --runs registered_<id> ...')
  name='Clean Bellman600 face-speed-accuracy '+' '.join(r.name for r in runs)
  config=dict(common,runs=[r.name for r in runs],speed_sha256={r.name:hashlib.sha256((r/'prepared/speed.pt').read_bytes()).hexdigest() for r in runs},samples=512 if a.smoke else 20000,sample_seed=20261018)
  for r in runs:inputs['speed_'+r.name]=str(r/'prepared/speed.pt')
  required=['accuracy/accuracy.json','accuracy/REPORT.md'];extra=['--runs',*[r.name for r in runs]]
 else:
  if not a.method or not a.cover:raise SystemExit('face-speed needs --method and --cover')
  name=f'Clean Bellman600 face-speed {a.method} {a.cover}'+(' separate-models' if a.method=='bellman' and a.speed_model=='separate' else '')+''.join(' '+v for v in variant)
  config=dict(common,method=a.method,cover=a.cover,speed_label='clearance only',data_seed=20261019,training_seed=87358)
  extra=['--method',a.method,'--cover',a.cover]
  if a.method=='metric' and not a.data:
   config.update(x0_filter=a.filter);extra+=['--filter',a.filter]
   if a.filter!='cover':name+=f' x0-filter-{a.filter}'
   for k in ('margin','offset','pairs'):
    v=getattr(a,k,None)
    if v is not None:config[k]=v;extra+=[f'--{k}',str(v)];name+=f' {k}-{v}'
  if a.method=='metric':
   config.update(source='ntrl-demo 5ed065f6040bac0003fc0cf69874e1bdfbc50a6a (vendored, unmodified trainer)',pairs=4096 if a.smoke else (a.pairs or 500000),epochs=2 if a.smoke else 5000,
    ntrl_mppi_seeds=[0] if a.smoke else [0,1,2])
   inputs.update(metric_train=str(CABINET/'metric/train.py'),metric_field=str(CABINET/'metric/field.py'),ntrl_demo=str(CABINET/'metric/ntrl_demo'))
   if a.data:
    data=HERE/'results'/Path(a.data).parts[-2]/'data' if Path(a.data).name=='data' else HERE/'results'/Path(a.data).name/'data'
    if not (data/'complete.json').exists():raise SystemExit(f'{data} has no complete metric data')
    config.update(data=str(data.relative_to(HERE)),data_reused=True);inputs['metric_data']=str(data);extra+=['--data',data.parent.name]
    if a.cover!='release':raise SystemExit('reused baseline data is the release label: use --cover release')
   elif a.cover=='release':raise SystemExit('--cover release needs --data (the baseline data)')
   required=['status.json','training/checkpoint.pt','training/progress.json','evaluation/evaluation.json','evaluation/REPORT.md']
  else:
   if not a.sampled:raise SystemExit('face-speed bellman needs --sampled registered_<id> (its sampled/ FK pool and coverage set)')
   sampled=HERE/'results'/Path(a.sampled).name/'sampled'
   if not (sampled/'complete.json').exists():raise SystemExit(f'{sampled} has no complete cabinet sampling')
   config.update(sampled=str(sampled.relative_to(HERE)),speed_seed=87358,bank_seed=87344,speed_steps=12 if a.smoke else a.speed_steps,bank_walkers=512 if a.smoke else 50000,
    bank_generations=2 if a.smoke else 8,losses=dict(bellman=6,eikonal=2,direction=2),rounds=2 if a.smoke else 5)
   inputs.update(sampled_inputs=str(sampled/'inputs'),sampled_manifest=str(sampled/'complete.json'),cabinet_prepare=str(CABINET/'prepare.py'),bank=str(HERE/'experiments/from_source/bank.py'),
    from_source_training=str(HERE/'experiments/from_source/training.py'))
   config.update(speed_model=a.speed_model)
   extra+=['--sampled',sampled.parent.name,'--speed-steps',str(a.speed_steps)]+(['--speed-model',a.speed_model] if a.speed_model!='shared' else [])
   required=['status.json','prepared/complete.json','prepared/speed.pt','prepared/bank/manifest.json','training/checkpoint.pt','evaluation/evaluation.json','evaluation/REPORT.md']
 spec=dict(name=name+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),config=config,environment=environment(),inputs=inputs,
  command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}','--gpus',gpus]+[x for v in variant for x in ['--scene',v]]+extra+(['--smoke'] if a.smoke else []),
  required_outputs=required,validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True)
 stem='_'.join([a.action]+variant+[x for x in extra if not x.startswith('--') and not x.isdigit()]).replace('.','p')
 path=directory/(stem+('_smoke' if a.smoke else '')+'.json');path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 subprocess.run([sys.executable,str(registry),'list','--search',a.action],check=True)
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
