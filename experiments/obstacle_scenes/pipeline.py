"""Registered from-source pipeline for one obstacle scene.

Same stages and recipes as experiments/transition_wall/pipeline.py. The endpoint audit and the
evaluator are the transition-wall scripts unchanged; their `wall_*` fields cover every added obstacle.
A last stage evaluates the same checkpoint on the scene's frozen certified test suites (tests/<scene>/).
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import sha,write
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
p.add_argument('--smoke',action='store_true');a=p.parse_args()
started=time.monotonic();timing={};smoke=['--smoke'] if a.smoke else []
# The audit only reads the fixed query endpoints, so it runs first and fails fast on a bad scene.
steps=[
 ('audit','/clean/experiments/transition_wall/audit.py',['--scene',a.scene,'--output',a.output/'audit']),
 ('sample','/clean/data_generation/generate.py',['--output',a.output/'sampled','--scene',a.scene]+smoke),
 ('prepare','/clean/experiments/from_source/prepare.py',['--inputs',a.output/'sampled/inputs','--output',a.output/'prepared','--scene',a.scene]+smoke),
 ('training','/clean/experiments/from_source/training.py',['--prepared',a.output/'prepared','--output',a.output/'training']+smoke),
 ('evaluate','/clean/experiments/transition_wall/evaluate.py',['--checkpoint',a.output/'training/checkpoint.pt','--output',a.output/'evaluation',
   '--scene',a.scene,'--endpoint-audit',a.output/'audit/endpoint_audit.json']+smoke),
 ('scene_tests','/clean/experiments/obstacle_scenes/evaluate_tests.py',['--checkpoint',a.output/'training/checkpoint.pt','--output',a.output/'scene_tests',
   '--scene',a.scene,'--tests',Path('/clean/experiments/obstacle_scenes/tests')/a.scene.stem]+smoke)]
for stage,script,extra in steps:
 write(a.output/'status.json',dict(complete=False,stage=stage,smoke=a.smoke,scene=str(a.scene),timing=timing))
 begin=time.monotonic()
 with (a.output/(stage+'.log')).open('w') as log:
  subprocess.run([sys.executable,'-u',script,*map(str,extra)],stdout=log,stderr=subprocess.STDOUT,check=True)
 timing[stage]=time.monotonic()-begin
result=json.loads((a.output/'evaluation/evaluation.json').read_text())
audit=json.loads((a.output/'audit/endpoint_audit.json').read_text())
tests=json.loads((a.output/'scene_tests/evaluation.json').read_text())
write(a.output/'status.json',dict(complete=True,stage='complete',smoke=a.smoke,scene=str(a.scene),scene_sha256=sha(a.scene),timing=timing,
 elapsed_seconds=time.monotonic()-started,mppi_successes=result.get('mppi_successes'),native_success=result.get('native_success'),
 endpoint_valid_queries=audit['wall_valid'],scene_tests_mppi_successes=tests['mppi_successes'],scene_tests_native_success=tests['native_success'],
 precomputed_training_assets_read=False,evaluation_data_fixed=True,mode='obstacle-scene-from-source'))
