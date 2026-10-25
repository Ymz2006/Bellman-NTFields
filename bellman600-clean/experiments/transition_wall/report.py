"""Compare the transition-wall runs with the no-wall from-source run from the same machine."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

TYPES=('cupping','pouring','cross')

def read(p):return json.loads(Path(p).read_text())

def flags(run,prefix):
    out={}
    for path in sorted((run/'evaluation').glob(prefix+'*.score.json')):
        label=path.name[:-len('.score.json')];out[label]=read(path)[label]['per_query']
    return out

def task_of(label):return label.split('_')[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('wall40',type=Path);p.add_argument('wall80',type=Path)
    p.add_argument('--no-wall',type=Path,required=True,help='no-wall from-source run from the same machine')
    p.add_argument('--output',type=Path,default=None);a=p.parse_args()
    root=a.root.resolve();out=(a.output or root/'evidence/transition_wall').resolve();out.mkdir(parents=True,exist_ok=True)
    suites=read(root/'runtime/tests/suite_manifest.json')['suites']
    pair_type={}
    for item in suites:
        with np.load(root/'runtime/tests/suites'/item['file'],allow_pickle=True) as z:pair_type[item['task']]=z['pair_type'].astype(int)
    runs={'no_wall':a.no_wall.resolve(),'wall40':a.wall40.resolve(),'wall80':a.wall80.resolve()}
    for name,run in runs.items():
        status=read(run/'status.json');assert status['complete'] and not status['smoke'],name
    audits={name:read(runs[name]/'audit/endpoint_audit.json') for name in ('wall40','wall80')}
    valid={name:{t:{r['query_id']:r['wall_endpoints_valid'] for r in v['per_query']} for t,v in audit['tasks'].items()} for name,audit in audits.items()}
    summary=dict(runs={k:str(v) for k,v in runs.items()},scenes={},results={})
    for name,audit in audits.items():
        blocked=Counter(TYPES[pair_type[t][r['query_id']]] for t,v in audit['tasks'].items() for r in v['per_query'] if not r['wall_endpoints_valid'])
        summary['scenes'][name]=dict(walls=audit['walls'],valid_endpoints=audit['wall_valid'],blocked_by_type=dict(blocked))
    for name,run in runs.items():
        status=read(run/'status.json');res=dict(status=status,methods={})
        for method,prefix,n in [('mppi','bellman_mppi_',1800),('rk4','bellman_rk4_',600)]:
            rows=flags(run,prefix);assert sum(len(v) for v in rows.values())==n,(name,method)
            ok=lambda subset=None:sum(v['success'] for l,qs in rows.items() for q,v in qs.items() if subset is None or valid[subset][task_of(l)][int(q)])
            count=lambda subset:sum(valid[subset][task_of(l)][int(q)] for l,qs in rows.items() for q in qs)
            by_type=Counter();by_type_n=Counter();by_task=Counter();reasons=Counter()
            for l,qs in rows.items():
                for q,v in qs.items():
                    kind=TYPES[pair_type[task_of(l)][int(q)]];by_type_n[kind]+=1;by_type[kind]+=bool(v['success']);by_task[task_of(l)]+=bool(v['success'])
                    if not v['success']:reasons.update(v['reasons'])
            res['methods'][method]=dict(n=n,success=ok(),subsets={s:dict(n=count(s),success=ok(s)) for s in valid},
                by_type={k:dict(success=by_type[k],n=by_type_n[k]) for k in TYPES},by_task=dict(by_task),failure_reasons=dict(reasons))
        prepared=run/'prepared'
        if (prepared/'bank/manifest.json').exists():
            bank=read(prepared/'bank/manifest.json');reports=bank['rollforward']['reports']
            res['bank']=dict(anchors=bank['anchors'],safe_edge_fraction=float(np.mean([r['safe_edge_fraction'] for r in reports])))
        if (prepared/'speed_audit.json').exists():res['speed_validation']=read(prepared/'speed_audit.json')['validation']
        if (run/'sampled/complete.json').exists():
            c=read(run/'sampled/complete.json');res['sampling']=dict(pool_size_each=c['pool_size_each'],original_pairs=c['original_pairs'],augmentation_pairs=c['augmentation_pairs'])
        summary['results'][name]=res
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')

    R=summary['results'];S=summary['scenes']
    def pct(k,n):return f'{k}/{n} ({100*k/n:.2f}%)' if n else '—'
    L=['# Transition-wall scenes: from-source retraining','',
       'All three runs used the same from-source pipeline, seeds and machine (RTX 3090, derived sm_86 image), so they differ only in the scene.','',
       '## Scenes','','|Scene|Wall x × y × z (m)|Centre (m)|Valid endpoints|Blocked: cupping / pouring / cross|','|---|---|---|---:|---|']
    for name,s in S.items():
        w=s['walls'][0];b=s['blocked_by_type']
        L.append(f"|{name}|{' × '.join(f'{d:.2f}' for d in w['dims'])}|({w['pose'][0]}, {w['pose'][1]}, {w['pose'][2]})|{s['valid_endpoints']}/600|{b.get('cupping',0)} / {b.get('pouring',0)} / {b.get('cross',0)}|")
    L+=['','## Success rates (all 600 queries)','','|Run|MPPI seeds 0/1/2|MPPI|RK4|','|---|---|---:|---:|']
    for name,r in R.items():
        L.append(f"|{name.replace('_','-')} ({Path(summary['runs'][name]).name})|{r['status']['mppi_successes']}|{pct(r['methods']['mppi']['success'],1800)}|{pct(r['methods']['rk4']['success'],600)}|")
    L+=['','## Same queries with and without the wall','',
        'Restricted to the queries whose endpoints are valid in each wall scene, so impossible queries do not count against the wall runs.','',
        '|Subset|Planner|No wall|With wall|','|---|---|---:|---:|']
    for s in ('wall40','wall80'):
        for m in ('mppi','rk4'):
            nw=R['no_wall']['methods'][m]['subsets'][s];ww=R[s]['methods'][m]['subsets'][s]
            L.append(f"|{s} valid ({S[s]['valid_endpoints']} queries)|{m.upper()}|{pct(nw['success'],nw['n'])}|{pct(ww['success'],ww['n'])}|")
    L+=['','## MPPI by query type','','The walls sit in the transition band, so cross queries (one side to the other) are the ones that must get past them.','',
        '|Query type|'+'|'.join(n.replace('_','-') for n in R)+'|','|---|'+'---:|'*len(R)]
    for k in TYPES:
        L.append(f'|{k}|'+'|'.join(pct(r['methods']['mppi']['by_type'][k]['success'],r['methods']['mppi']['by_type'][k]['n']) for r in R.values())+'|')
    L+=['','## MPPI by task (/300)','','|Task|'+'|'.join(n.replace('_','-') for n in R)+'|','|---|'+'---:|'*len(R)]
    for t in R['no_wall']['methods']['mppi']['by_task']:
        L.append(f'|{t}|'+'|'.join(str(r['methods']['mppi']['by_task'][t]) for r in R.values())+'|')
    L+=['','## Failure reasons (a run can have several)','','|Run|MPPI|RK4|','|---|---|---|']
    for name,r in R.items():
        f=lambda m:', '.join(f'{k} {v}' for k,v in sorted(r['methods'][m]['failure_reasons'].items(),key=lambda x:-x[1]))
        L.append(f"|{name.replace('_','-')}|{f('mppi')}|{f('rk4')}|")
    L+=['','## Data, training and timing','','|Run|Bank rows|Bank safe-move fraction|Speed log-RMSE (orig / aug)|Stage seconds|Total|','|---|---:|---:|---|---|---:|']
    for name,r in R.items():
        b=r.get('bank',{});speed=' / '.join('%.3f'%v['log_rmse'] for v in r.get('speed_validation',[]))
        L.append(f"|{name.replace('_','-')}|{b.get('anchors','—')}|{b.get('safe_edge_fraction',float('nan')):.3f}|{speed}|"
                 f"{', '.join(f'{k} {v:.0f}' for k,v in r['status']['timing'].items())}|{r['status']['elapsed_seconds']/60:.1f} min|")
    L+=['','## Caveats','',
        '- Modified environment (cuRobo rebuilt for sm_86, host CUDA driver libraries): independent samples, not bitwise replays of the original machine.',
        '- One training seed per scene; the 600 queries are a reused development set. No significance or generalization claims.',
        '- T(q,g) takes only q and g. Each wall reaches it only through the scene-specific training data, speed labels and bank.',
        '- Valid endpoints do not prove a collision-free path exists, especially for cross queries past the 0.80 m wall.',
        '- The frozen no-wall baseline and committed evidence are unchanged.']
    (out/'COMPARISON.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))

if __name__=='__main__':main()
