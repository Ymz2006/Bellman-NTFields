"""Register metric NTFields (ntrl-demo) cabinet runs with preprocessing / planner settings (experiments/ntrl_cabinet/README.md).

    run.py ntrl-cabinet [--scene cabinet_w25] [--data registered_<id>] --ntrl-args "--variant metric_arm --coords ntrl ..." --gpu 1

--ntrl-args is passed verbatim to experiments/ntrl_cabinet/pipeline.py (data.py, train.py and evaluate.py options);
--data reuses a registered ntrl-cabinet run's data/ (its coordinates must match).
"""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
from launch import HERE,RELEASE
from tools.runtime_assets import verify
from experiments.cabinet.run import scene_paths
from experiments.face_speed.run import environment

EXPERIMENT=HERE/'experiments/ntrl_cabinet';FILES=['data.py','train.py','field.py','evaluate.py','pipeline.py','run.py']
NTRL=HERE/'baselines/ntrl-demo';NTRL_FILES=['models/data_mlp.py']+[f'models/{v}/{f}' for v in ('metric','metric_arm') for f in ('model_train_metric.py','model_network_metric.py','model_function_metric.py')]
DEPENDS=['experiments/cabinet/geometry.py','experiments/transition_wall/evaluate.py','data_generation/source','method']


def execute(a):
 verify()
 registry=HERE/'tools/experiment_registry/registry.py';gpus='all' if a.gpu is None else f'device={a.gpu}'
 scene=a.scene or 'cabinet';SCENE,TESTS=scene_paths(scene);variant=[] if scene=='cabinet' else [scene]
 args=shlex.split(a.ntrl_args or '');tests_only='--make-tests' in args
 if not tests_only and not (TESTS/'suite_manifest.json').exists():raise SystemExit(f'Freeze the {scene} test suites into {TESTS} first (run.py cabinet-tests, freeze_tests.py).')
 if '--data' in args or '--output' in args or '--scene' in args or '--tests' in args:raise SystemExit('--ntrl-args may not set --data/--output/--scene/--tests')
 data=HERE/'results'/Path(a.data).name/'data' if a.data else None
 if data and not (data/'complete.json').exists():raise SystemExit(f'{data} has no complete data')
 inputs=dict(scene=str(SCENE),**({} if tests_only else dict(scene_tests=str(TESTS))),method=str(HERE/'method'),runtime=str(RELEASE),launcher=str(HERE/'launch.py'),entrypoint=str(HERE/'run.py'),
  runtime_manifest=str(HERE/'runtime_manifest.json'),registry=str(registry),**{'ntrl_cabinet_'+f:str(EXPERIMENT/f) for f in FILES},
  **{'ntrl_demo_'+f.replace('/','_'):str(NTRL/f) for f in NTRL_FILES},**{'depends_'+f.replace('/','_'):str(HERE/f) for f in DEPENDS},
  **(dict(tests_wrapper=str(EXPERIMENT/'make_tests_pi.py'),generator=str(HERE/'experiments/cabinet/make_tests.py'),obstacle_generator=str(HERE/'experiments/obstacle_scenes/make_tests.py')) if tests_only else {}))
 if data:inputs.update(data_manifest=str(data/'complete.json'),data_points=str(data/'sampled_points.npy'))
 head=(NTRL/'.git/HEAD').read_text().strip();commit=(NTRL/'.git'/head[5:]).read_text().strip() if head.startswith('ref: ') else head
 config=dict(action=a.action,method='metric_ntfields',scene=scene,scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(),smoke=a.smoke,gpus=gpus,
  ntrl_demo_commit=commit,pipeline_args=args,data=str(data.relative_to(HERE)) if data else None,planning_seeds=[0] if a.smoke else [0,1,2],scorer_tilt_deg=180,
  defaults='data.py/train.py/evaluate.py defaults = the registered cabinet-metric recipe')
 extra=(['--data',data.parent.name] if data else [])+['--ntrl-args='+shlex.join(args)]
 spec=dict(name='Clean Bellman600 ntrl-cabinet '+' '.join(variant+([data.parent.name] if data else [])+[shlex.join(args) or 'defaults'])+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),
  config=config,environment=environment(),inputs=inputs,
  command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}','--gpus',gpus]+[x for v in variant for x in ['--scene',v]]+extra+(['--smoke'] if a.smoke else []),
  required_outputs=['status.json','tests/suite_manifest.json','tests/generation.json','tests/suites/*.npz'] if tests_only else
   ['status.json','training/checkpoint.pt','training/progress.json','evaluation/evaluation.json','evaluation/REPORT.md'],validation='files')
 tag=hashlib.sha256(json.dumps([scene,a.data,args,a.smoke]).encode()).hexdigest()[:10]
 directory=HERE/'specs';directory.mkdir(exist_ok=True);path=directory/f'ntrl-cabinet_{scene}_{tag}{"_smoke" if a.smoke else ""}.json'
 path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
