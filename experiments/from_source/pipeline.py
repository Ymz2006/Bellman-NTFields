"""Registered end-to-end training with no precomputed training artifacts."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import write
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
started=time.monotonic();timing={};steps=[
 ('sample','/clean/data_generation/generate.py',['--output',a.output/'sampled']),
 ('prepare','/clean/experiments/from_source/prepare.py',['--inputs',a.output/'sampled/inputs','--output',a.output/'prepared'])]
if not a.prepare_only:steps += [
 ('training','/clean/experiments/from_source/training.py',['--prepared',a.output/'prepared','--output',a.output/'training']),
 ('evaluate','/clean/method/evaluate.py',['--checkpoint',a.output/'training/checkpoint.pt','--output',a.output/'evaluation','--no-reference'])]
for stage,script,extra in steps:
 write(a.output/'status.json',dict(complete=False,stage=stage,smoke=a.smoke,timing=timing))
 begin=time.monotonic()
 with (a.output/(stage+'.log')).open('w') as log:
  subprocess.run([sys.executable,'-u',script,*map(str,extra)]+(['--smoke'] if a.smoke else []),stdout=log,stderr=subprocess.STDOUT,check=True)
 timing[stage]=time.monotonic()-begin
result={} if a.prepare_only else json.loads((a.output/'evaluation/evaluation.json').read_text())
write(a.output/'status.json',dict(complete=True,stage='complete',smoke=a.smoke,timing=timing,elapsed_seconds=time.monotonic()-started,
 mppi_successes=result.get('mppi_successes'),native_success=result.get('native_success'),precomputed_training_assets_read=False,
 original_pose_pool_read=False,evaluation_data_fixed=True,mode='from-source'))
