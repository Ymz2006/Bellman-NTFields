"""Run the clean five-round training and unchanged 600-query scoring."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import write
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
for stage,extra in [('train',['--output',a.output/'training']),('evaluate',['--checkpoint',a.output/'training/checkpoint.pt','--output',a.output/'evaluation'])]:
 write(a.output/'status.json',dict(complete=False,stage=stage,smoke=a.smoke))
 with (a.output/(stage+'.log')).open('w') as log:
  subprocess.run([sys.executable,'-u','/clean/method/'+('training' if stage=='train' else stage)+'.py',*map(str,extra)]+(['--smoke'] if a.smoke else []),stdout=log,stderr=subprocess.STDOUT,check=True)
e=json.loads((a.output/'evaluation/evaluation.json').read_text())
assert e['complete']
write(a.output/'status.json',dict(complete=True,stage='complete',smoke=a.smoke,successes=e['mppi_successes'],native_success=e['native_success']))
