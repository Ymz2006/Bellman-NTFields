"""Compare wall-trained checkpoint against frozen no-wall transfer on identical queries."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('run',type=Path);a=p.parse_args()
 root=a.root.resolve();run=a.run.resolve();evaluation=read(run/'evaluation/evaluation.json');status=read(run/'status.json')
 assert status['complete'] and not status['smoke'] and evaluation['complete']
 transfer=root/'results/registered_20260930T003519Z-f50d55a8/evaluation'
 original=root/'evidence/baseline/evaluation';audit=evaluation['endpoint_audit'];rows={};tasks={}
 for method,pattern,expected in [('mppi','bellman_mppi_*.score.json',1800),('rk4','bellman_rk4_*.score.json',600)]:
  c=dict(n=0,eligible_n=0,no_wall_success=0,transfer_success=0,wall_trained_success=0,
         no_wall_eligible=0,transfer_eligible=0,wall_trained_eligible=0,rescued=0,regressed=0,
         wall_affected_executions=0,wall_affected_rescued=0)
  diffs=[];failures=Counter()
  for path in sorted((run/'evaluation').glob(pattern)):
   label=path.name.removesuffix('.score.json');task=label.split('_')[2]
   new=read(path)[label]['per_query'];old=read(transfer/path.name)[label]['per_query'];nw=read(original/path.name)[label]['per_query']
   eligible={str(q['query_id']):q['wall_endpoints_valid'] for q in audit['tasks'][task]['per_query']}
   assert new.keys()==old.keys()==nw.keys()
   t=tasks.setdefault(task,{}).setdefault(method,dict(n=0,transfer_success=0,wall_trained_success=0,eligible_n=0))
   for q in new:
    x,y,z=bool(new[q]['success']),bool(old[q]['success']),bool(nw[q]['success']);valid=bool(eligible[q])
    assert not x or valid
    c['n']+=1;c['eligible_n']+=valid;c['no_wall_success']+=z;c['transfer_success']+=y;c['wall_trained_success']+=x
    c['no_wall_eligible']+=valid and z;c['transfer_eligible']+=valid and y;c['wall_trained_eligible']+=valid and x
    c['rescued']+=x and not y;c['regressed']+=y and not x
    affected=valid and z and not y;c['wall_affected_executions']+=affected;c['wall_affected_rescued']+=affected and x
    t['n']+=1;t['eligible_n']+=valid;t['transfer_success']+=y;t['wall_trained_success']+=x
    if x!=y:diffs.append(dict(label=label,query_id=int(q),transfer_success=y,wall_trained_success=x,eligible=valid,old_reasons=old[q]['reasons'],new_reasons=new[q]['reasons']))
    if not x:failures.update(new[q]['reasons'])
  assert c['n']==expected
  c.update(failure_reason_counts=dict(failures),differences=diffs);rows[method]=c
 model=read(run/'training/config.json');bank=read(run/'prepared/bank/manifest.json')
 summary=dict(run_id=run.name,checkpoint_sha256=hashlib.sha256((run/'training/checkpoint.pt').read_bytes()).hexdigest(),
  mppi_successes=evaluation['mppi_successes'],native_success=evaluation['native_success'],status=status,
  training=model,bank=dict(anchors=bank['anchors'],actions_per_anchor=bank['actions_per_anchor'],elapsed_seconds=bank['elapsed_seconds']),
  speed_audit=read(run/'prepared/speed_audit.json'),physical_label_audit=read(run/'prepared/physical_label_audit.json'),
  comparisons=rows,tasks=tasks)
 (run/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Wall25: retraining speed, offline bank and TravelCost','',
  f'Official run: `{run.name}`. Round 5 is fixed; the checkpoint was not selected by evaluation.','',
  '|Method|No-wall training → wall evaluation|Wall training → wall evaluation|Valid-endpoint subset: no-wall training|Valid-endpoint subset: wall training|',
  '|---|---:|---:|---:|---:|']
 for method,c in rows.items():
  def cell(k,den):return f'{c[k]}/{c[den]} ({100*c[k]/c[den]:.2f}%)'
  lines.append(f'|{method.upper()}|{cell("transfer_success","n")}|{cell("wall_trained_success","n")}|{cell("transfer_eligible","eligible_n")}|{cell("wall_trained_eligible","eligible_n")}|')
 lines+=['',f'MPPI for the three planning seeds: {evaluation["mppi_successes"]}. 600 distinct queries, of which 28 have a start or goal blocked by the wall and 572 have valid endpoints; the three planning seeds are not three training seeds.','',
  '|Task|Direct transfer /300|Wall retraining /300|Difference|','|---|---:|---:|---:|']
 for task,t in tasks.items():
  c=t['mppi'];lines.append(f'|{task}|{c["transfer_success"]}|{c["wall_trained_success"]}|{c["wall_trained_success"]-c["transfer_success"]:+d}|')
 lines+=['','## Paired changes','']
 for method,c in rows.items():
  lines.append(f'- {method.upper()}: {c["rescued"]} rescued, {c["regressed"]} new failures. Of the {c["wall_affected_executions"]} executions that succeeded without the wall, failed with it and have valid endpoints, retraining rescued {c["wall_affected_rescued"]}.')
 lines+=['','## Training and timing','',
  f'- speed: 20000 steps, fresh initialization; an independent training pose pool was relabeled, and no evaluation queries were read.',
  f'- Offline bank: {bank["anchors"]} source poses × 32 candidates, generated in {bank["elapsed_seconds"]:.2f} seconds.',
  f'- TravelCost: 338177 parameters, losses 6/2/2, 5 rounds × 2 epochs, {model["expected_updates"]} steps.',
  f'- Stage timing: {json.dumps(status["timing"])}; total {status["elapsed_seconds"]:.2f} seconds.',
  '', 'Training and evaluation use a fixed wall; wall information enters supervision through speed and bank, and the model still takes only q,g as input. This is not generalization to arbitrary new walls. Valid endpoints do not prove that a path exists. Single training seed and a reused development query set; no significance or independent-generalization claims. The no-wall baseline was not replaced.']
 (run/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(mppi_successes=evaluation['mppi_successes'],native_success=evaluation['native_success'],bank_anchors=bank['anchors'],updates=model['expected_updates'],timing=status['timing']),indent=2))
 for method,c in rows.items():print(method,json.dumps({k:v for k,v in c.items() if k!='differences'}))
if __name__=='__main__':main()
