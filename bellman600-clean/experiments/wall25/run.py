"""Registered wall evaluation using the clean repository's frozen interfaces."""
import json,subprocess,sys
from pathlib import Path
from launch import HERE,RELEASE,IMAGE

def execute(a):
 if a.smoke and a.action=='wall-audit':raise SystemExit('Endpoint audit is already a small check; omit --smoke')
 registry=HERE/'tools/experiment_registry/registry.py'
 required=['endpoint_audit.json'] if a.action=='wall-audit' else ['evaluation/evaluation.json','evaluation/REPORT.md']
 spec=dict(name='Clean Bellman600 Wall25 '+a.action+(' smoke' if a.smoke else ' full'),kind='evaluation',root=str(HERE),
   config=dict(action=a.action,smoke=a.smoke,training=False,checkpoint='frozen_nowall_baseline',
       planning_seeds=[0] if a.smoke else [0,1,2],scene='wall25_overlap010',queries=2 if a.smoke else 600),
   environment=dict(docker_context='default',docker_image_id=IMAGE),
   inputs=dict(experiment=str(HERE/'experiments/wall25'),method=str(HERE/'method'),launcher=str(HERE/'launch.py'),
       checkpoint=str(HERE/'assets/checkpoint.pt'),reference=str(HERE/'assets/reference'),runtime=str(RELEASE),entrypoint=str(HERE/'run.py')),
   command=[sys.executable,str(HERE/'launch.py'),a.action,'--output','{output}']+(['--smoke'] if a.smoke else []),
   required_outputs=required,validation='files')
 directory=HERE/'specs';directory.mkdir(exist_ok=True);path=directory/(a.action+('_smoke' if a.smoke else '')+'.json')
 path.write_text(json.dumps(spec,indent=2)+'\n')
 if a.prepare_only:print(path);return
 subprocess.run([sys.executable,str(registry),'list','--search','Bellman600 Wall25'],check=True)
 check=subprocess.run([sys.executable,str(registry),'check',str(path)],capture_output=True,text=True)
 (directory/(path.stem+'.check.json')).write_text(check.stdout)
 if check.returncode and not (check.returncode==3 and a.repeat_reason):raise SystemExit(check.stderr or 'Registry check blocked; inspect specs/'+path.stem+'.check.json')
 command=[sys.executable,str(registry),'run',str(path)]
 if a.repeat_reason:command+=['--repeat-reason',a.repeat_reason]
 subprocess.run(command,check=True)
