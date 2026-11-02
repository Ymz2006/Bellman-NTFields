"""Register P-NTFields cabinet runs (experiments/pntfields_cabinet/README.md).

    run.py pntfields-cabinet [--scene cabinet_w25] [--data registered_<id>] [--pnt-args "--joint-limit 3.14159"] --gpu 0

--pnt-args is passed verbatim to experiments/pntfields_cabinet/pipeline.py; --data reuses a registered
pntfields-cabinet run's data/ (its coordinates must match).
"""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
from launch import HERE,RELEASE
from tools.runtime_assets import verify
from experiments.cabinet.run import scene_paths
from experiments.face_speed.run import environment

EXPERIMENT=HERE/'experiments/pntfields_cabinet';FILES=['field.py','train.py','evaluate.py','pipeline.py','run.py']
PNT=HERE/'baselines/P-NTFields';PNT_FILES=['models/model_res_sigmoid.py','models/data_mlp.py']
DEPENDS=['experiments/ntrl_cabinet/data.py','experiments/ntrl_cabinet/field.py','experiments/cabinet/geometry.py','experiments/transition_wall/evaluate.py','data_generation/source','method']


def execute(a):
 verify()
 registry=HERE/'tools/experiment_registry/registry.py';gpus='all' if a.gpu is None else f'device={a.gpu}'
 scene=a.scene or 'cabinet';SCENE,TESTS=scene_paths(scene);variant=[] if scene=='cabinet' else [scene]
 args=shlex.split(a.pnt_args or '')
 if not (TESTS/'suite_manifest.json').exists():raise SystemExit(f'Freeze the {scene} test suites into {TESTS} first.')
 if '--data' in args or '--output' in args or '--scene' in args or '--tests' in args:raise SystemExit('--pnt-args may not set --data/--output/--scene/--tests')
 data=HERE/'results'/Path(a.data).name/'data' if a.data else None
 if data and not (data/'complete.json').exists():raise SystemExit(f'{data} has no complete data')
 inputs=dict(scene=str(SCENE),scene_tests=str(TESTS),method=str(HERE/'method'),runtime=str(RELEASE),launcher=str(HERE/'launch.py'),entrypoint=str(HERE/'run.py'),
  runtime_manifest=str(HERE/'runtime_manifest.json'),registry=str(registry),**{'pntfields_cabinet_'+f:str(EXPERIMENT/f) for f in FILES},
  **{'pntfields_'+f.replace('/','_'):str(PNT/f) for f in PNT_FILES},**{'depends_'+f.replace('/','_'):str(HERE/f) for f in DEPENDS})
 if data:inputs.update(data_manifest=str(data/'complete.json'),data_points=str(data/'sampled_points.npy'))
 head=(PNT/'.git/HEAD').read_text().strip();commit=(PNT/'.git'/head[5:]).read_text().strip() if head.startswith('ref: ') and (PNT/'.git'/head[5:]).exists() else head
 config=dict(action=a.action,method='pntfields',scene=scene,scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(),smoke=a.smoke,gpus=gpus,
  pntfields_commit=commit,pipeline_args=args,data=str(data.relative_to(HERE)) if data else None,planning_seeds=[0] if a.smoke else [0,1,2],scorer_tilt_deg=180,
  defaults='pipeline.py defaults = P-NTFields arm recipe (2M pairs, margin 0.5/12 m, ntrl coordinates, 10,000 epochs)')
 extra=(['--data',data.parent.name] if data else [])+['--pnt-args='+shlex.join(args)]
 spec=dict(name='Clean Bellman600 pntfields-cabinet '+' '.join(variant+([data.parent.name] if data else [])+[shlex.join(args) or 'defaults'])+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),
  config=config,environment=environment(),inputs=inputs,
  command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}','--gpus',gpus]+[x for v in variant for x in ['--scene',v]]+extra+(['--smoke'] if a.smoke else []),
  required_outputs=['status.json','training/checkpoint.pt','training/progress.json','evaluation/evaluation.json','evaluation/REPORT.md'],validation='files')
 tag=hashlib.sha256(json.dumps([scene,a.data,args,a.smoke]).encode()).hexdigest()[:10]
 directory=HERE/'specs';directory.mkdir(exist_ok=True);path=directory/f'pntfields-cabinet_{scene}_{tag}{"_smoke" if a.smoke else ""}.json'
 path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
