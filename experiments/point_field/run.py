"""Register the hierarchical point-field experiment (experiments/point_field/README.md).

point-field --sampled registered_<id> --runs registered_<id> ... [--self-net registered_<split run>]
  Trains the point-distance fields on the cabinet scene and compares the hierarchical clearance/speed with the
  speed networks of the given runs (prepared/speed.pt). Read-only on those runs; no planning.
point-field-bellman --speed learned|exact --run registered_<point-field run> --sampled registered_<id> --reference registered_<baseline>
  The cabinet Bellman pipeline with the hierarchical speed (bellman.py): learned = the point-field run's pool-only
  field, exact = the release label itself. Poses, seed states and bank transitions must equal the baseline's.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
from launch import HERE,RELEASE
from tools.runtime_assets import verify
from experiments.cabinet.run import scene_paths
from experiments.face_speed.run import environment

EXPERIMENT=HERE/'experiments/point_field';FILES=['point_field.py','train.py','compare.py','pipeline.py','run.py']
BELLMAN_FILES=['point_field.py','bellman.py','bellman_pipeline.py','run.py']
BELLMAN_DEPENDS=['experiments/face_speed/robot.py','experiments/face_speed/clearance.py','experiments/face_speed/training_pieces.py','experiments/face_speed/speed_net.py',
 'experiments/from_source/bank.py','experiments/cabinet/prepare.py','experiments/cabinet/geometry.py','experiments/cabinet/evaluate.py','experiments/transition_wall/evaluate.py','method']
DEPENDS=['experiments/face_speed/robot.py','experiments/face_speed/clearance.py','experiments/face_speed/speed_net.py','experiments/face_speed/speed_accuracy.py','method/speed.py','method/common.py']


def execute(a):
 if a.action=='point-field-bellman':return execute_bellman(a)
 verify()
 registry=HERE/'tools/experiment_registry/registry.py';gpus='all' if a.gpu is None else f'device={a.gpu}'
 scene=a.scene or 'cabinet';SCENE,TESTS=scene_paths(scene);variant=[] if scene=='cabinet' else [scene]
 if not a.sampled:raise SystemExit('point-field needs --sampled registered_<id> (a cabinet run with sampled/)')
 sampled=HERE/'results'/Path(a.sampled).name/'sampled'
 if not (sampled/'complete.json').exists():raise SystemExit(f'{sampled} has no complete cabinet sampling')
 runs=[HERE/'results'/Path(r).name for r in (a.runs or [])];self_net=HERE/'results'/Path(a.self_net).name if a.self_net else None
 for r in runs+([self_net] if self_net else []):
  if not (r/'prepared/speed.pt').exists():raise SystemExit(f'{r.name} has no prepared/speed.pt')
 inputs=dict(scene=str(SCENE),scene_tests=str(TESTS),method=str(HERE/'method'),runtime=str(RELEASE),launcher=str(HERE/'launch.py'),entrypoint=str(HERE/'run.py'),
  runtime_manifest=str(HERE/'runtime_manifest.json'),registry=str(registry),sampled_inputs=str(sampled/'inputs'),sampled_manifest=str(sampled/'complete.json'),
  **{'point_field_'+f:str(EXPERIMENT/f) for f in FILES},**{'depends_'+Path(f).name:str(HERE/f) for f in DEPENDS},
  **{'speed_'+r.name:str(r/'prepared/speed.pt') for r in runs+([self_net] if self_net else [])})
 steps=a.speed_steps
 config=dict(action=a.action,scene=scene,scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(),smoke=a.smoke,gpus=gpus,sampled=str(sampled.relative_to(HERE)),
  runs=[r.name for r in runs],self_net=self_net.name if self_net else None,speed_sha256={r.name:hashlib.sha256((r/'prepared/speed.pt').read_bytes()).hexdigest() for r in runs+([self_net] if self_net else [])},
  field=dict(width=256,depth=5,features=128,sigma=8.,seed=20261025,steps=50 if a.smoke else steps,batch=65536,sources=['pool+uniform','pool']),samples=512 if a.smoke else 20000,sample_seed=20261018)
 extra=['--sampled',sampled.parent.name,'--speed-steps',str(steps)]+(['--runs',*[r.name for r in runs]] if runs else [])+(['--self-net',self_net.name] if self_net else [])
 spec=dict(name='Clean Bellman600 point-field '+' '.join(variant+[r.name for r in runs])+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),config=config,environment=environment(),inputs=inputs,
  command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}','--gpus',gpus]+[x for v in variant for x in ['--scene',v]]+extra+(['--smoke'] if a.smoke else []),
  required_outputs=['status.json','fields/pool_uniform/field.pt','fields/pool/field.pt','comparison/comparison.json','comparison/REPORT.md'],validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True);path=directory/('point-field'+''.join('_'+v for v in variant)+('_smoke' if a.smoke else '')+'.json')
 path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)


def execute_bellman(a):
 verify()
 registry=HERE/'tools/experiment_registry/registry.py';gpus='all' if a.gpu is None else f'device={a.gpu}'
 scene=a.scene or 'cabinet';SCENE,TESTS=scene_paths(scene);variant=[] if scene=='cabinet' else [scene]
 if a.speed not in ('learned','exact'):raise SystemExit('point-field-bellman needs --speed learned|exact')
 if not a.sampled or not a.reference:raise SystemExit('point-field-bellman needs --sampled and --reference (the baseline rerun whose seed states and bank it must match)')
 sampled=HERE/'results'/Path(a.sampled).name/'sampled';reference=HERE/'results'/Path(a.reference).name/'prepared'
 if not (sampled/'complete.json').exists():raise SystemExit(f'{sampled} has no complete cabinet sampling')
 if not (reference/'bank/manifest.json').exists():raise SystemExit(f'{reference} has no bank')
 inputs=dict(scene=str(SCENE),scene_tests=str(TESTS),runtime=str(RELEASE),launcher=str(HERE/'launch.py'),entrypoint=str(HERE/'run.py'),
  runtime_manifest=str(HERE/'runtime_manifest.json'),registry=str(registry),sampled_inputs=str(sampled/'inputs'),sampled_manifest=str(sampled/'complete.json'),
  reference_seed_states=str(reference/'seed_states.npy'),reference_bank_manifest=str(reference/'bank/manifest.json'),
  **{'point_field_'+f:str(EXPERIMENT/f) for f in BELLMAN_FILES},**{'depends_'+f.replace('/','_'):str(HERE/f) for f in BELLMAN_DEPENDS})
 config=dict(action=a.action,scene=scene,scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(),smoke=a.smoke,gpus=gpus,speed=a.speed,sampled=str(sampled.relative_to(HERE)),
  reference=str(reference.relative_to(HERE)),bank_seed=87344,bank_walkers=512 if a.smoke else 50000,bank_generations=2 if a.smoke else 8,losses=dict(bellman=6,eikonal=2,direction=2),
  training_seed=87358,rounds=2 if a.smoke else 5,planning_seeds=[0] if a.smoke else [0,1,2],scene_tests=str(TESTS.relative_to(HERE)),scorer_tilt_deg=180)
 extra=['--speed',a.speed,'--sampled',sampled.parent.name,'--reference',reference.parent.name]
 if a.speed=='learned':
  if not a.run:raise SystemExit('--speed learned needs --run registered_<point-field run>')
  field=HERE/'results'/Path(a.run).name/'fields/pool/field.pt'
  if not field.exists():raise SystemExit(f'{field} missing')
  inputs['field']=str(field);config.update(field_run=field.parents[2].name,field_sha256=hashlib.sha256(field.read_bytes()).hexdigest());extra+=['--run',field.parents[2].name]
 spec=dict(name=f'Clean Bellman600 point-field-bellman {a.speed}'+''.join(' '+v for v in variant)+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),config=config,environment=environment(),inputs=inputs,
  command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}','--gpus',gpus]+[x for v in variant for x in ['--scene',v]]+extra+(['--smoke'] if a.smoke else []),
  required_outputs=['status.json','prepared/complete.json','prepared/speed.pt','prepared/bank/manifest.json','training/checkpoint.pt','evaluation/evaluation.json','evaluation/REPORT.md'],validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True);path=directory/(f'point-field-bellman_{a.speed}'+''.join('_'+v for v in variant)+('_smoke' if a.smoke else '')+'.json')
 path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
