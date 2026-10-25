"""Registered cabinet-task pipeline for one method, ending on the frozen cabinet test suites.

bellman: generate.py (FK pool, clearance labels, coverage) -> prepare.py (speed network, offline bank)
         -> experiments/from_source/training.py (unchanged 338,177-parameter TravelCost, losses 6/2/2,
         seed 87358, 5 rounds) -> evaluate.py on the round-5 checkpoint.
metric:  metric/data.py (ntrl-demo arm recipe in the cabinet box) -> metric/train.py (ntrl-demo
         trainer, 5,000 epochs) -> evaluate.py on the final epoch.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import sha,write
HERE='/clean/experiments/cabinet'
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
p.add_argument('--tests',type=Path,required=True);p.add_argument('--method',choices=['bellman','metric'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
started=time.monotonic();timing={};smoke=['--smoke'] if a.smoke else [];o=a.output
evaluate=(f'{HERE}/evaluate.py',['--method',a.method,'--checkpoint',o/'training/checkpoint.pt','--scene',a.scene,'--tests',a.tests,'--output',o/'evaluation']+smoke)
if a.method=='bellman':steps=[
 ('sample',f'{HERE}/generate.py',['--output',o/'sampled','--scene',a.scene]+smoke),
 ('prepare',f'{HERE}/prepare.py',['--inputs',o/'sampled/inputs','--output',o/'prepared','--scene',a.scene]+smoke),
 ('training','/clean/experiments/from_source/training.py',['--prepared',o/'prepared','--output',o/'training']+smoke),('evaluate',*evaluate)]
else:steps=[
 ('data',f'{HERE}/metric/data.py',['--scene',a.scene,'--output',o/'data']+smoke),
 ('training',f'{HERE}/metric/train.py',['--data',o/'data','--output',o/'training']+smoke),('evaluate',*evaluate)]
for stage,script,extra in steps:
 write(o/'status.json',dict(complete=False,stage=stage,method=a.method,smoke=a.smoke,scene=str(a.scene),timing=timing))
 begin=time.monotonic()
 with (o/(stage+'.log')).open('w') as log:
  subprocess.run([sys.executable,'-u',script,*map(str,extra)],stdout=log,stderr=subprocess.STDOUT,check=True)
 timing[stage]=time.monotonic()-begin
result=json.loads((o/'evaluation/evaluation.json').read_text())
write(o/'status.json',dict(complete=True,stage='complete',method=a.method,smoke=a.smoke,scene=str(a.scene),scene_sha256=sha(a.scene),
 tests_manifest_sha256=sha(a.tests/'suite_manifest.json'),timing=timing,elapsed_seconds=time.monotonic()-started,totals=result['totals'],
 mppi_successes=result['mppi_successes'],native_success=result['native_success'],mode=f'cabinet-{a.method}'))
