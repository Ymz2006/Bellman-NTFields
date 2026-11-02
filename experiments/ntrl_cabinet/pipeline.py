"""metric NTFields (ntrl-demo) on a cabinet scene: data.py -> train.py -> evaluate.py.

--checkpoint evaluates an existing train.py checkpoint (planner settings only; no data or training).
Every option is forwarded to its stage; with the defaults the pipeline is the registered cabinet-metric
recipe (release coordinates and label, models/metric, 5,000 epochs). --data reuses an earlier data/ folder.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import sha,write
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--tests',type=Path,required=True)
p.add_argument('--data',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--coords',default='release');p.add_argument('--label',default='release')
p.add_argument('--margin',type=float,default=.05);p.add_argument('--offset',type=float,default=.005);p.add_argument('--pairs',type=int,default=500000)
p.add_argument('--window',type=float,nargs=2);p.add_argument('--step',type=float,default=.5);p.add_argument('--data-seed',type=int,default=20261019)
p.add_argument('--joint-limit',type=float);p.add_argument('--focus',type=float,default=0.);p.add_argument('--focus-sigma',type=float,default=.15)
p.add_argument('--variant',default='metric');p.add_argument('--epochs',type=int,default=5000);p.add_argument('--train-seed',type=int,default=87358)
p.add_argument('--td-step',type=float);p.add_argument('--raw-speed',action='store_true');p.add_argument('--metric-scale',type=float);p.add_argument('--weight-decay',type=float);p.add_argument('--lr',type=float);p.add_argument('--normal-weight',type=float);p.add_argument('--td-weight',type=float)
p.add_argument('--ntrl',nargs='*',default=[],help='evaluate.py --ntrl-* options, e.g. steps=400 samples=200');p.add_argument('--no-release',action='store_true')
p.add_argument('--make-tests',action='store_true',help='only generate the scene\'s certified suites in its joint_limit domain (make_tests_pi.py) into tests/')
p.add_argument('--smoke',action='store_true');a=p.parse_args()
if a.make_tests:
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'status.json',dict(complete=False,stage='tests'))
    with (a.output/'tests.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'make_tests_pi.py'),'--scene',str(a.scene),'--output',str(a.output/'tests')]+(['--smoke'] if a.smoke else []),stdout=log,stderr=subprocess.STDOUT,check=True)
    write(a.output/'status.json',dict(complete=True,stage='complete',mode='ntrl-cabinet tests',scene_sha256=sha(a.scene)));sys.exit(0)
started=time.monotonic();timing={};smoke=['--smoke'] if a.smoke else [];o=a.output;o.mkdir(parents=True,exist_ok=True)
data=a.data or o/'data'
steps=[] if a.data or a.checkpoint else [('data',HERE/'data.py',['--scene',a.scene,'--output',data,'--coords',a.coords,'--label',a.label,'--margin',a.margin,'--offset',a.offset,
    '--pairs',a.pairs,'--step',a.step,'--seed',a.data_seed,'--focus',a.focus,'--focus-sigma',a.focus_sigma]+(['--joint-limit',a.joint_limit] if a.joint_limit else [])+(['--window',*a.window] if a.window else [])+smoke)]
if a.data:
    done=json.loads((data/'complete.json').read_text());assert done['complete'] and done['coords']==a.coords,f'{data}: coords {done["coords"]} != {a.coords}'
checkpoint=a.checkpoint or o/'training/checkpoint.pt'
if not a.checkpoint:steps+=[('training',HERE/'train.py',['--data',data,'--output',o/'training','--variant',a.variant,'--epochs',a.epochs,'--seed',a.train_seed]
    +(['--td-step',a.td_step] if a.td_step is not None else [])+(['--normal-weight',a.normal_weight] if a.normal_weight is not None else [])+(['--td-weight',a.td_weight] if a.td_weight is not None else [])+(['--raw-speed'] if a.raw_speed else [])+(['--metric-scale',a.metric_scale] if a.metric_scale is not None else [])
    +(['--weight-decay',a.weight_decay] if a.weight_decay is not None else [])+(['--lr',a.lr] if a.lr is not None else [])+smoke)]
steps+=[('evaluate',HERE/'evaluate.py',['--checkpoint',checkpoint,'--variant',a.variant,'--coords',a.coords,'--scene',a.scene,'--tests',a.tests,'--output',o/'evaluation']
     +sum((['--ntrl-'+kv.split('=')[0],kv.split('=')[1]] for kv in a.ntrl),[])+(['--no-release'] if a.no_release else [])+(['--metric-scale',a.metric_scale] if a.metric_scale is not None else [])+smoke)]
settings={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
for stage,script,extra in steps:
    write(o/'status.json',dict(complete=False,stage=stage,settings=settings,timing=timing))
    begin=time.monotonic()
    with (o/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-u',str(script),*map(str,extra)],stdout=log,stderr=subprocess.STDOUT,check=True)
    timing[stage]=time.monotonic()-begin
result=json.loads((o/'evaluation/evaluation.json').read_text())
write(o/'status.json',dict(complete=True,stage='complete',method='metric',settings=settings,scene_sha256=sha(a.scene),tests_manifest_sha256=sha(a.tests/'suite_manifest.json'),
    timing=timing,elapsed_seconds=time.monotonic()-started,totals=result['totals'],mode='ntrl-cabinet'))
