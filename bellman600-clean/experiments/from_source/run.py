"""Register independent data generation/training; only frozen runtime is required."""
import json,subprocess,sys
from launch import HERE,RELEASE,IMAGE
from tools.runtime_assets import verify

def execute(a):
 verify()
 if a.action=='doctor-source':
  subprocess.run(['docker','--context','default','image','inspect',IMAGE,'--format','{{.Id}}'],check=True)
  print('Frozen runtime verified. No bank, speed, checkpoint, or historical training inputs required.');return
 prepare=a.action=='from-source-prepare';registry=HERE/'tools/experiment_registry/registry.py'
 required=['status.json','sampled/complete.json','sampled/inputs/source_manifest.json','prepared/complete.json','prepared/speed.pt','prepared/bank/manifest.json']
 if not prepare:required+=['training/checkpoint.pt','evaluation/evaluation.json']
 spec=dict(name='Clean Bellman600 independent '+a.action+(' smoke' if a.smoke else ' full'),kind='pipeline',root=str(HERE),
 config=dict(action=a.action,smoke=a.smoke,fk_seed=42,pair_seed=42,coverage_seed=20260912,speed_seed=87358,bank_seed=87344,training_seed=87358,
 fk_pool_each=4096 if a.smoke else 2000000,speed_steps=12 if a.smoke else 20000,bank_walkers=512 if a.smoke else 50000,
 bank_generations=2 if a.smoke else 8,losses=dict(bellman=6,eikonal=2,direction=2),rounds=2 if a.smoke else 5,
 precomputed_training_inputs=False,planning_seeds=[0] if a.smoke else [0,1,2]),
 environment=dict(docker_context='default',docker_image_id=IMAGE),
 inputs=dict(data_generation=str(HERE/'data_generation'),method=str(HERE/'method'),experiment=str(HERE/'experiments/from_source'),
 runtime=str(RELEASE),launcher=str(HERE/'launch.py'),entrypoint=str(HERE/'run.py'),runtime_manifest=str(HERE/'runtime_manifest.json'),
 runtime_tools=str(HERE/'tools/runtime_assets.py'),registry=str(registry)),
 command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}']+(['--smoke'] if a.smoke else []),
 required_outputs=required,validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True);path=directory/(a.action+('_smoke' if a.smoke else '')+'.json');path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 subprocess.run([sys.executable,str(registry),'list','--search','Clean Bellman600 independent'],check=True)
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True);(directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not(check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
