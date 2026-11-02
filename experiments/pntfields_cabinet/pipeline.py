"""P-NTFields on a cabinet scene: ntrl_cabinet/data.py (P-NTFields recipe) -> train.py -> evaluate.py.

Data: P-NTFields' arm recipe (dataprocessing/speed_sampling_gpu.py, configs/arm.txt) expressed with the
ntrl_cabinet generator: 2,000,000 pairs, x0 with clearance offset < d < margin (margin = 0.5/12 m, offset =
margin/10), x1 = x0 + unit(rand - 0.5) * U(0, sqrt 3) inside the joint domain, speed clip(d, offset, margin) /
margin, robot-obstacle distance only (no self term), coordinates x = q / (2 pi) (P-NTFields: scale = pi / 0.5).
Adapted to the task as for the ntrl-demo comparison: x0's grasp point lies in the scene's sampling box and d is
the exact sphere-cuboid clearance. --joint-limit restricts the joints (P-NTFields samples |q| <= pi).
--data reuses an earlier data/ folder; --checkpoint evaluates an existing checkpoint.
"""
import argparse,json,math,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import sha,write
HERE=Path(__file__).resolve().parent;DATA=Path('/clean/experiments/ntrl_cabinet/data.py')
MARGIN=.5/12;OFFSET=MARGIN/10
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--tests',type=Path,required=True)
p.add_argument('--data',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--coords',default='ntrl');p.add_argument('--pairs',type=int,default=2000000)
p.add_argument('--data-seed',type=int,default=20261026);p.add_argument('--joint-limit',type=float);p.add_argument('--epochs',type=int,default=10000);p.add_argument('--train-seed',type=int,default=87358)
p.add_argument('--beta',type=float,default=.015);p.add_argument('--dg',type=float,default=.03);p.add_argument('--no-release',action='store_true');p.add_argument('--no-ntrl',action='store_true')
p.add_argument('--smoke',action='store_true');a=p.parse_args()
started=time.monotonic();timing={};smoke=['--smoke'] if a.smoke else [];o=a.output;o.mkdir(parents=True,exist_ok=True)
data=a.data or o/'data'
steps=[] if a.data or a.checkpoint else [('data',DATA,['--scene',a.scene,'--output',data,'--coords',a.coords,'--label','env','--margin',MARGIN,'--offset',OFFSET,
    '--window',OFFSET-.01,MARGIN-.01,'--pairs',a.pairs,'--step',math.sqrt(3),'--seed',a.data_seed]+(['--joint-limit',a.joint_limit] if a.joint_limit else [])+smoke)]
if a.data:
    done=json.loads((data/'complete.json').read_text());assert done['complete'] and done['coords']==a.coords,f'{data}: coords {done["coords"]} != {a.coords}'
checkpoint=a.checkpoint or o/'training/checkpoint.pt'
if not a.checkpoint:steps+=[('training',HERE/'train.py',['--data',data,'--output',o/'training','--epochs',a.epochs,'--seed',a.train_seed]+smoke)]
steps+=[('evaluate',HERE/'evaluate.py',['--checkpoint',checkpoint,'--coords',a.coords,'--scene',a.scene,'--tests',a.tests,'--output',o/'evaluation','--beta',a.beta,'--dg',a.dg]
     +(['--no-release'] if a.no_release else [])+(['--no-ntrl'] if a.no_ntrl else [])+smoke)]
settings={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
for stage,script,extra in steps:
    write(o/'status.json',dict(complete=False,stage=stage,settings=settings,timing=timing))
    begin=time.monotonic()
    with (o/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-u',str(script),*map(str,extra)],stdout=log,stderr=subprocess.STDOUT,check=True)
    timing[stage]=time.monotonic()-begin
result=json.loads((o/'evaluation/evaluation.json').read_text())
write(o/'status.json',dict(complete=True,stage='complete',method='pntfields',settings=settings,scene_sha256=sha(a.scene),tests_manifest_sha256=sha(a.tests/'suite_manifest.json'),
    timing=timing,elapsed_seconds=time.monotonic()-started,totals=result['totals'],mode='pntfields-cabinet'))
