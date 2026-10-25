"""Compare rebuilt numeric inputs with the frozen no-wall baseline."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import ROOT,write,sha
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);a=p.parse_args()
old=torch.load(ROOT/'assets/speed.pt',map_location='cpu',weights_only=False)['model_state_dict']
new=torch.load(a.prepared/'speed.pt',map_location='cpu',weights_only=False)['model_state_dict']
assert old.keys()==new.keys()
weights={k:dict(equal=torch.equal(old[k],new[k]),max_absolute_error=float((old[k]-new[k]).abs().max())) for k in old}
rows={}
for path in sorted((ROOT/'assets/bank').glob('*.npy')):
 rebuilt=a.prepared/'bank'/path.name
 # exact_safe is an alias of safe, kept by the historical builder.
 if path.name=='exact_safe.npy' and not rebuilt.exists():rebuilt=a.prepared/'bank/safe.npy'
 x=np.load(path,mmap_mode='r');y=np.load(rebuilt,mmap_mode='r')
 equal=x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y)
 rows[path.name]=dict(equal=equal,original_sha256=sha(path),rebuilt_sha256=sha(rebuilt),shape=list(y.shape))
result=dict(speed_parameters_equal=all(v['equal'] for v in weights.values()),speed_parameters=weights,bank_arrays_equal=all(v['equal'] for v in rows.values()),bank_arrays=rows)
write(a.prepared/'rebuild_validation.json',result)
print(json.dumps({k:v for k,v in result.items() if k.endswith('_equal')}),flush=True)
if not result['speed_parameters_equal'] or not result['bank_arrays_equal']:raise SystemExit('Rebuild differs from frozen baseline; inspect rebuild_validation.json')
