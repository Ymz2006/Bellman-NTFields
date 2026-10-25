"""No-wall training inputs -> speed -> bank -> fixed-round model -> strict scoring."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import write
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
started=time.monotonic();timing={}
steps=[('prepare',['--output',a.output/'prepared']),('training',['--prepared',a.output/'prepared','--output',a.output/'training']),
       ('evaluate',['--checkpoint',a.output/'training/checkpoint.pt','--output',a.output/'evaluation'])]
for stage,extra in steps:
    write(a.output/'status.json',dict(complete=False,stage=stage,smoke=a.smoke,timing=timing))
    begin=time.monotonic()
    with (a.output/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-u',('/clean/method/evaluate.py' if stage=='evaluate' else f'/clean/experiments/no_wall_rebuild/{stage}.py'),*map(str,extra)]+(['--smoke'] if a.smoke else []),stdout=log,stderr=subprocess.STDOUT,check=True)
    timing[stage]=time.monotonic()-begin
    if stage=='prepare' and not a.smoke:
        subprocess.run([sys.executable,'/clean/experiments/no_wall_rebuild/verify.py','--prepared',str(a.output/'prepared')],check=True)
result=json.loads((a.output/'evaluation/evaluation.json').read_text());assert result['complete']
write(a.output/'status.json',dict(complete=True,stage='complete',smoke=a.smoke,timing=timing,elapsed_seconds=time.monotonic()-started,
    mppi_successes=result['mppi_successes'],native_success=result['native_success']))
