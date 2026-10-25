"""Test a run's speed network against ground-truth speed on sampled configurations.

Ground truth is PhysicalSpeed (experiments/from_source/physical.py): the sphere-cuboid obstacle distance,
self-collision distance and cup/pour tilt that produced the training labels, in the run's own scene.
The prediction is the run's SpeedField (prepared/speed.pt). Sample sets:
  uniform               configurations drawn uniformly over the training domain |q_norm| <= 0.5
  uniform_valid         the same draw kept only where the release gate accepts it (collision-free, tilt OK)
  heldout_<source>      the run's validation endpoints (pair index modulo 100 == 0), never fitted
Checks: speeds stay inside [0.1, 1]; the checkpoint was trained in this scene; the held-out task-speed
error reproduces the run's own prepared/speed_audit.json. Read-only; no training and no planning.
"""
import sys
sys.path.insert(0,'/clean/method');sys.path.insert(0,'/clean/experiments/from_source')
import argparse,json
from pathlib import Path
from common import RELEASE,sha,write
from speed import load_speed
from geometry import ExactLocalGate
from physical import PhysicalSpeed
import numpy as np
import torch

SAMPLES=20000
SEED=20261018
FLOOR=.1+1e-4
# The floor is 0.1 up to float32 rounding, so the first bin starts at 0.
BINS=[(0.,.2),(.2,.5),(.5,.9),(.9,1.)]
AUDIT_TOLERANCE=dict(mae=1e-3,p95=2e-3)


def components(net,q):
    return net.head(net.tower(net.features(q))).clamp(-2.302585093,0).exp()


@torch.no_grad()
def compare(net,physical,q,chunk=4096):
    """Ground truth and prediction, both as the scalar speed and as the two task speeds."""
    out=dict(true=[],pred=[],true_tasks=[],pred_tasks=[],valid=[])
    for begin in range(0,len(q),chunk):
        x=torch.as_tensor(q[begin:begin+chunk],device='cuda:0')
        # Copy each result off the GPU before the next cuRobo call reuses its buffers.
        out['true'].append(physical(x).cpu().numpy())
        out['true_tasks'].append(physical.tasks(x)[0].cpu().numpy())
        out['valid'].append(physical.gate.states(x).cpu().numpy())
        out['pred'].append(net(x).cpu().numpy())
        out['pred_tasks'].append(components(net,x).cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}


def metrics(true,pred):
    if not len(true):return dict(n=0)
    e=pred-true;a=np.abs(e)
    row=dict(n=int(len(a)),mae=float(a.mean()),rmse=float(np.sqrt((e**2).mean())),bias=float(e.mean()),p50=float(np.median(a)),
             p95=float(np.quantile(a,.95)),maximum=float(a.max()),log_rmse=float(np.sqrt(((np.log(pred)-np.log(np.maximum(true,.1)))**2).mean())),
             pearson=float(np.corrcoef(true,pred)[0,1]) if true.std()>0 and pred.std()>0 else None)
    row['by_true_speed']=[]
    for lo,hi in BINS:
        m=(true>=lo)&((true<hi) if hi<1 else (true<=hi+1e-5))
        row['by_true_speed'].append(dict(low=lo,high=hi,n=int(m.sum()),mae=float(a[m].mean()) if m.any() else None,bias=float(e[m].mean()) if m.any() else None))
    return row


def summarize(r):
    true,pred,valid=r['true'],r['pred'],r['valid'].astype(bool)
    task_error=np.abs(r['pred_tasks']-r['true_tasks'])
    blocked,free=true<=FLOOR,true>=.9
    return dict(speed=metrics(true,pred),valid_configurations=metrics(true[valid],pred[valid]),invalid_configurations=metrics(true[~valid],pred[~valid]),
                share_valid=float(valid.mean()),share_true_at_floor=float(blocked.mean()),
                task_speeds=dict(endpoint_count=int(len(task_error)),mae=float(task_error.mean()),p95=float(np.quantile(task_error,.95))),
                floor_detection=dict(true_at_floor=int(blocked.sum()),predicted_below_half=float((pred[blocked]<.5).mean()) if blocked.any() else None,
                                     true_free=int(free.sum()),predicted_at_least_half=float((pred[free]>=.5).mean()) if free.any() else None))


def report_md(report):
    f=lambda v,d=3:'–' if v is None else f'{v:.{d}f}'
    lines=[f"# Speed network test: {report['run']}",'',f"Scene `{Path(report['scene']).name}`. Checkpoint sha256 `{report['speed_sha256'][:12]}`."+(' SMOKE ONLY.' if report['smoke'] else ''),'',
           'Ground truth is PhysicalSpeed, the geometry that made the training labels. Speeds lie in [0.1, 1]; 0.1 is the floor (in contact, within 5 mm, or tilted past the limit).','',
           '|Set|Points|MAE|RMSE|p95|Max|Bias|Pearson|Valid configs|True speed at floor|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in report['sets'].items():
        m=s['speed'];lines.append(f"|{name}|{m['n']}|{f(m['mae'])}|{f(m['rmse'])}|{f(m['p95'])}|{f(m['maximum'])}|{f(m['bias'],4)}|{f(m['pearson'])}|{100*s['share_valid']:.1f}%|{100*s['share_true_at_floor']:.1f}%|")
    lines+=['','MAE by true speed:','','|Set|'+'|'.join(f"{max(lo,.1)}–{hi}" for lo,hi in BINS)+'|','|---|'+'---:|'*len(BINS)]
    for name,s in report['sets'].items():
        lines.append(f'|{name}|'+'|'.join(f"{f(b['mae'])} (n={b['n']})" for b in s['speed']['by_true_speed'])+'|')
    lines+=['','|Set|Valid configs MAE|Invalid configs MAE|At-floor points predicted < 0.5|Free points (≥ 0.9) predicted ≥ 0.5|','|---|---:|---:|---:|---:|']
    for name,s in report['sets'].items():
        d=s['floor_detection'];pct=lambda v:'–' if v is None else f'{100*v:.1f}%'
        lines.append(f"|{name}|{f(s['valid_configurations'].get('mae'))}|{f(s['invalid_configurations'].get('mae'))}|{pct(d['predicted_below_half'])}|{pct(d['predicted_at_least_half'])}|")
    lines+=['','## Checks','']+[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['name']}. {c['detail']}" for c in report['checks']]
    return '\n'.join(lines)+'\n'


@torch.no_grad()
def valid_uniform(gate,rng,count,chunk=65536):
    kept=[];total=0
    while total<count:
        q=rng.uniform(-.5,.5,(chunk,6)).astype(np.float32)
        ok=np.concatenate([gate.states(torch.as_tensor(q[b:b+4096],device='cuda:0')).cpu().numpy() for b in range(0,chunk,4096)])
        kept.append(q[ok]);total+=int(ok.sum())
        if len(kept)>2000:raise RuntimeError('Too few valid configurations in the uniform draw')
    return np.concatenate(kept)[:count]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,help='results/registered_<id> directory name with prepared/speed.pt')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run=Path('/clean/results')/a.run;a.output.mkdir(parents=True,exist_ok=False)
    status=json.loads((run/'status.json').read_text())
    scene=Path(status['scene']) if status.get('scene') else RELEASE/'configs/scene.yml'
    speed_path=run/'prepared/speed.pt';inputs=run/'sampled/inputs';audit_path=run/'prepared/speed_audit.json'
    contract=torch.load(speed_path,map_location='cpu',weights_only=False).get('training_contract',{})
    net=load_speed(speed_path,'cuda:0');physical=PhysicalSpeed(ExactLocalGate(scene))
    count=512 if a.smoke else SAMPLES
    rng=np.random.default_rng(SEED)
    sets={'uniform':rng.uniform(-.5,.5,(count,6)).astype(np.float32),'uniform_valid':valid_uniform(physical.gate,rng,count)}
    if inputs.exists():
        for kind in ['original','augmentation']:
            points=np.load(inputs/kind/'sampled_points.npy',mmap_mode='r');ids=np.arange(len(points));ids=ids[ids%100==0]
            sets['heldout_'+kind]=np.asarray(points[ids[:256] if a.smoke else ids],dtype=np.float32).reshape(-1,6)
    report=dict(run=a.run,speed=str(speed_path),speed_sha256=sha(speed_path),scene=str(scene),scene_sha256=sha(scene),seed=SEED,smoke=a.smoke,sets={},checks=[])
    arrays={}
    for name,q in sets.items():
        r=compare(net,physical,q);arrays[name]=dict(q=q,**r);report['sets'][name]=summarize(r);print('SPEED_TEST',name,report['sets'][name]['speed']['mae'],flush=True)
    low=min(min(float(v['true'].min()),float(v['pred'].min())) for v in arrays.values());high=max(max(float(v['true'].max()),float(v['pred'].max())) for v in arrays.values())
    report['checks'].append(dict(name='speeds inside [0.1, 1]',passed=low>=.1-1e-5 and high<=1+1e-5,detail=f'observed {low:.4f} to {high:.4f}.'))
    trained=contract.get('scene_sha256')
    report['checks'].append(dict(name='checkpoint trained in this scene',passed=trained==report['scene_sha256'],detail=f"training contract {str(trained)[:12]}, scene {report['scene_sha256'][:12]}."))
    if audit_path.exists() and not a.smoke and inputs.exists():
        for row in json.loads(audit_path.read_text())['validation']:
            ours=report['sets']['heldout_'+row['source']]['task_speeds']
            ok=all(abs(ours[k]-row[k])<=AUDIT_TOLERANCE[k] for k in AUDIT_TOLERANCE)
            report['checks'].append(dict(name=f"held-out {row['source']} task-speed error reproduces speed_audit.json",passed=ok,
                detail=f"MAE {ours['mae']:.5f} vs {row['mae']:.5f}, p95 {ours['p95']:.5f} vs {row['p95']:.5f} (tolerance {AUDIT_TOLERANCE['mae']}/{AUDIT_TOLERANCE['p95']})."))
    report['passed']=all(c['passed'] for c in report['checks'])
    np.savez_compressed(a.output/'samples.npz',**{f'{name}_{k}':v for name,r in arrays.items() for k,v in r.items()})
    write(a.output/'speed_test.json',report);(a.output/'REPORT.md').write_text(report_md(report))
    print('SPEED_TEST_PASSED' if report['passed'] else 'SPEED_TEST_FAILED',flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
