"""Summarize wall transfer, matched eligible subsets and saved-path identity."""
import argparse,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('run',type=Path);a=p.parse_args()
 root=a.root.resolve();run=a.run.resolve();wall=run/'evaluation';base=root/'evidence/baseline/evaluation'
 report=read(wall/'evaluation.json');assert report['complete'] and not report['smoke']
 audit=report['endpoint_audit'];comparisons={};tasks={}
 for method,pattern,n in [('mppi','bellman_mppi_*.score.json',1800),('rk4','bellman_rk4_*.score.json',600)]:
  counters=dict(n=0,baseline_success=0,wall_success=0,eligible_n=0,baseline_eligible_success=0,wall_eligible_success=0,
      rescued=0,regressed=0,regressed_invalid_endpoints=0,regressed_valid_endpoints=0,identical_paths=0)
  differences=[]
  for path in sorted(wall.glob(pattern)):
   label=path.name.removesuffix('.score.json');b=read(base/path.name)[label]['per_query'];w=read(path)[label]['per_query']
   task=label.split('_')[2];eligible={str(r['query_id']):r['wall_endpoints_valid'] for r in audit['tasks'][task]['per_query']}
   old_paths=np.load(base/(label+'.npz'),allow_pickle=True)['trajectories']
   new_paths=np.load(wall/(label+'.npz'),allow_pickle=True)['trajectories']
   assert len(old_paths)==len(new_paths)==100 and b.keys()==w.keys()
   row=tasks.setdefault(task,{}).setdefault(method,dict(n=0,baseline_success=0,wall_success=0,eligible_n=0,baseline_eligible_success=0,wall_eligible_success=0))
   for key in w:
    old,new,valid=bool(b[key]['success']),bool(w[key]['success']),bool(eligible[key]);i=int(key)
    counters['identical_paths']+=int(np.array_equal(old_paths[i],new_paths[i]))
    for stats in [counters,row]:
     stats['n']+=1;stats['baseline_success']+=old;stats['wall_success']+=new;stats['eligible_n']+=valid
     stats['baseline_eligible_success']+=old and valid;stats['wall_eligible_success']+=new and valid
    if old!=new:
     counters['rescued']+=int(new);counters['regressed']+=int(old)
     counters['regressed_invalid_endpoints']+=int(old and not valid)
     counters['regressed_valid_endpoints']+=int(old and valid)
     differences.append(dict(label=label,query_id=i,old_success=old,new_success=new,endpoints_valid=valid,reasons=w[key]['reasons']))
  assert counters['n']==n
  counters['differences']=differences;comparisons[method]=counters
 summary=dict(run_id=run.name,wall=audit['wall'],invalid_endpoint_queries=600-audit['wall_valid'],valid_endpoint_queries=audit['wall_valid'],
              mppi_seeds=report['mppi_successes'],comparisons=comparisons,tasks=tasks,
              interpretation='Frozen no-wall checkpoint, unchanged planner. Direct transfer only; no scene input and no new wall training. Endpoint validity alone does not certify route feasibility.')
 (run/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Wall25: direct transfer of the frozen no-wall checkpoint','',f'Run: `{run.name}`. No retraining and no planner parameter changes; only a 25 cm wall is added.','',
 '|Method|No wall, all 600 queries|Wall, all 600 queries|No wall, valid-endpoint subset|Wall, valid-endpoint subset|','|---|---:|---:|---:|---:|']
 for method,c in comparisons.items():
  def cell(k,den):return f'{c[k]}/{c[den]} ({100*c[k]/c[den]:.2f}%)'
  lines.append(f'|{method.upper()}|{cell("baseline_success","n")}|{cell("wall_success","n")}|{cell("baseline_eligible_success","eligible_n")}|{cell("wall_eligible_success","eligible_n")}|')
 lines+=['',f'MPPI successes for the three planning seeds: {report["mppi_successes"]}. There are 600 distinct queries; {audit["wall_valid"]} have valid endpoints, and the added wall invalidates the start or goal of {600-audit["wall_valid"]} queries.','',
 '|Task|MPPI no wall /300|MPPI wall /300|Valid-endpoint queries /100|','|---|---:|---:|---:|']
 for task,row in tasks.items():
  c=row['mppi'];lines.append(f'|{task}|{c["baseline_success"]}|{c["wall_success"]}|{audit["tasks"][task]["wall_valid"]}|')
 lines+=['','## Saved trajectories and changes in failures','']
 for method,c in comparisons.items():
  lines.append(f'- {method.upper()}: saved trajectories identical point for point in {c["identical_paths"]}/{c["n"]}; {c["regressed"]} new failures, {c["rescued"]} rescued. Of the new failures, {c["regressed_invalid_endpoints"]} have invalid endpoints and {c["regressed_valid_endpoints"]} still have valid endpoints.')
 lines+=['','T(q,g) receives no obstacle information, and the default MPPI does not enable geometric safety re-ranking. This result must not be read as a success rate after training with the wall, nor as evidence of adaptation to new obstacles. The valid-endpoint subset has no full path-feasibility certificate. The frozen no-wall baseline remains at 88.72% and was not replaced.','',
 'The scene keeps the same normalization as the original 600 queries, a ±0.10 m transition zone, the 20-degree OR constraint, 0.005/0.002 m collision margins and 0.001 rad dense scoring; this is not the historical Suite-safe180 protocol.']
 (run/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ('comparisons','tasks')},indent=2))
 for method,c in comparisons.items():print(method,json.dumps({k:v for k,v in c.items() if k!='differences'}))
if __name__=='__main__':main()
