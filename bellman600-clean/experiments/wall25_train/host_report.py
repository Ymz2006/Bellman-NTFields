"""Compare a new Wall25 retraining run with the recorded original-machine evidence.

Per-query flags for the original machine are reconstructed from committed evidence:
  no wall           evidence/baseline/evaluation/*.score.json
  wall transfer     no wall, flipped at evidence/wall25/comparison.json differences
                    (saved paths were identical and nothing was rescued)
  wall retraining   transfer, flipped at evidence/wall25_train/comparison.json differences
Each reconstruction is checked against the recorded totals before it is used.
Writes host_comparison.json and HOST_COMPARISON.md into the run directory.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def label_of(path):return path.name[:-len('.score.json')]

def flags(folder,pattern):
    out={}
    for path in sorted(Path(folder).glob(pattern)):
        label=label_of(path);out[label]={int(k):v for k,v in read(path)[label]['per_query'].items()}
    return out

def flipped(base,differences,key):
    out={label:{q:bool(v['success'] if isinstance(v,dict) else v) for q,v in rows.items()} for label,rows in base.items()}
    for d in differences:out[d['label']][int(d['query_id'])]=bool(d[key])
    return out

def total(rows,valid=None):
    return sum(ok for label,qs in rows.items() for q,ok in qs.items() if valid is None or valid[label][q])

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('run',type=Path);a=p.parse_args()
    root=a.root.resolve();run=a.run.resolve()
    status=read(run/'status.json');evaluation=read(run/'evaluation/evaluation.json')
    assert status['complete'] and not status['smoke'] and evaluation['complete']
    audit=read(root/'experiments/wall25/endpoint_audit.json')
    wall=read(root/'evidence/wall25/comparison.json');trained=read(root/'evidence/wall25_train/comparison.json')
    summary=dict(run_id=run.name,checkpoint_sha256=hashlib.sha256((run/'training/checkpoint.pt').read_bytes()).hexdigest(),
        mppi_successes=evaluation['mppi_successes'],native_success=evaluation['native_success'],status=status,methods={})
    for method,pattern,n,recorded in [('mppi','bellman_mppi_*.score.json',1800,(1597,1494,1475)),('rk4','bellman_rk4_*.score.json',600,(549,508,494))]:
        base=flags(root/'evidence/baseline/evaluation',pattern);new=flags(run/'evaluation',pattern)
        assert base.keys()==new.keys() and sum(len(v) for v in new.values())==n
        valid={label:{r['query_id']:bool(r['wall_endpoints_valid']) for r in audit['tasks'][label.split('_')[2]]['per_query']} for label in base}
        no_wall={label:{q:bool(v['success']) for q,v in rows.items()} for label,rows in base.items()}
        transfer=flipped(no_wall,wall['comparisons'][method]['differences'],'new_success')
        original=flipped(transfer,trained['comparisons'][method]['differences'],'wall_trained_success')
        here={label:{q:bool(v['success']) for q,v in rows.items()} for label,rows in new.items()}
        assert (total(no_wall),total(transfer),total(original))==recorded,'reconstruction does not reproduce recorded totals'
        columns=dict(no_wall=no_wall,transfer=transfer,original_retrain=original,host_retrain=here)
        tasks={}
        for label in sorted(base):
            row=tasks.setdefault(label.split('_')[2],{k:0 for k in columns})
            for k,v in columns.items():row[k]+=sum(v[label].values())
        reasons=Counter(r for rows in new.values() for v in rows.values() if not v['success'] for r in v['reasons'])
        def paired(a_,b_):
            return dict(rescued=sum(b_[l][q] and not a_[l][q] for l in a_ for q in a_[l]),
                        regressed=sum(a_[l][q] and not b_[l][q] for l in a_ for q in a_[l]))
        agree=sum(here[l][q]==original[l][q] for l in here for q in here[l])
        summary['methods'][method]=dict(n=n,eligible_n=total({l:{q:True for q in v} for l,v in valid.items()},valid),
            totals={k:total(v) for k,v in columns.items()},eligible_totals={k:total(v,valid) for k,v in columns.items()},
            tasks=tasks,failure_reason_counts=dict(reasons),
            host_vs_transfer=paired(transfer,here),host_vs_original_retrain=paired(original,here),
            flag_agreement_with_original_retrain=agree)
    prepared=run/'prepared'
    for name,path in [('speed_audit','speed_audit.json'),('physical_label_audit','physical_label_audit.json'),('bank','bank/manifest.json')]:
        if (prepared/path).exists():summary[name]=read(prepared/path)
    if (run/'environment_override.json').exists():summary['environment_override']=read(run/'environment_override.json')
    (run/'host_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')

    def pct(k,n):return f'{k}/{n} ({100*k/n:.2f}%)'
    lines=['# Wall25 retraining on this machine vs recorded original-machine results','',
        f'Run: `{run.name}`. Round 5 is fixed; the checkpoint was not selected by evaluation.',
        f"Checkpoint SHA256: `{summary['checkpoint_sha256']}`.",'',
        '"Original" columns are reconstructed per query from committed evidence of the original RTX 4090 runs; they were not re-run here.','',
        '## Success rates','',
        '|Planner|No wall (original)|Wall transfer (original)|Wall retrain (original)|**Wall retrain (this machine)**|',
        '|---|---:|---:|---:|---:|']
    for method,m in summary['methods'].items():
        t=m['totals'];lines.append(f"|{method.upper()}|{pct(t['no_wall'],m['n'])}|{pct(t['transfer'],m['n'])}|{pct(t['original_retrain'],m['n'])}|**{pct(t['host_retrain'],m['n'])}**|")
    lines+=['','Valid-endpoint subset (572 queries whose start and goal are not blocked by the wall):','',
        '|Planner|No wall (original)|Wall transfer (original)|Wall retrain (original)|**Wall retrain (this machine)**|','|---|---:|---:|---:|---:|']
    for method,m in summary['methods'].items():
        t=m['eligible_totals'];lines.append(f"|{method.upper()}|{pct(t['no_wall'],m['eligible_n'])}|{pct(t['transfer'],m['eligible_n'])}|{pct(t['original_retrain'],m['eligible_n'])}|**{pct(t['host_retrain'],m['eligible_n'])}**|")
    lines+=['',f"MPPI per planning seed on this machine: {summary['mppi_successes']} (original retrain: {trained['mppi_successes']}). "
        'The three planning seeds are not three training seeds.','',
        '## MPPI by task (/300)','','|Task|No wall|Wall transfer|Wall retrain (original)|Wall retrain (this machine)|','|---|---:|---:|---:|---:|']
    for task,r in summary['methods']['mppi']['tasks'].items():
        lines.append(f"|{task}|{r['no_wall']}|{r['transfer']}|{r['original_retrain']}|{r['host_retrain']}|")
    lines+=['','## Paired changes','']
    for method,m in summary['methods'].items():
        a_,b_=m['host_vs_transfer'],m['host_vs_original_retrain']
        lines.append(f"- {method.upper()}: versus wall transfer, this run rescues {a_['rescued']} and newly fails {a_['regressed']}. "
            f"Versus the original retrain, it rescues {b_['rescued']} and newly fails {b_['regressed']}; "
            f"{m['flag_agreement_with_original_retrain']}/{m['n']} per-query flags agree.")
    lines+=['','## Failure reasons on this machine','','|Planner|Reasons (a run can have several)|','|---|---|']
    for method,m in summary['methods'].items():
        lines.append(f"|{method.upper()}|{', '.join(f'{k} {v}' for k,v in sorted(m['failure_reason_counts'].items(),key=lambda x:-x[1]))}|")
    lines+=['','## Training and timing','']
    if 'physical_label_audit' in summary:
        lines.append('- Relabeling: '+'; '.join(f"{s['source']} {s['wall_changed_pairs']}/{s['pairs']} pairs slowed by the wall" for s in summary['physical_label_audit']['sources'])+'.')
    if 'speed_audit' in summary:
        lines.append('- Speed network validation log-RMSE: '+', '.join(f"{v['source']} {v['log_rmse']:.4f}" for v in summary['speed_audit']['validation'])+'.')
    if 'bank' in summary:
        lines.append(f"- Offline bank: {summary['bank']['anchors']} source poses × 32 candidates (original machine: {trained['bank']['anchors']}).")
    lines.append(f"- Stage timing: {json.dumps({k:round(v,1) for k,v in status['timing'].items()})}; total {status['elapsed_seconds']:.1f} s "
        f"(original machine: {trained['status']['elapsed_seconds']:.1f} s).")
    lines+=['','## Caveats','',
        '- This machine uses a modified environment (cuRobo rebuilt for sm_86, host CUDA driver libraries); see `environment_override.json`. '
        'Results are an independent sample, not a bitwise replay of the original run.',
        '- One training seed and a reused development query set: no significance or generalization claims.',
        '- T(q,g) still takes only q and g; the wall reaches it only through relabeled speeds and the bank. Valid endpoints do not prove a path exists.',
        '- The no-wall baseline and committed evidence are unchanged.']
    (run/'HOST_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
