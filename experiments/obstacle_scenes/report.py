"""Compare obstacle-scene runs with the no-wall from-source run from the same machine."""
import argparse,hashlib,json
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
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('runs',type=Path,nargs='+',help='obstacle-scene runs (results/registered_<id>)')
    p.add_argument('--no-wall',type=Path,required=True,help='no-wall from-source run from the same machine')
    p.add_argument('--tests-eval',type=Path,nargs='*',default=[],help='obstacle-evaluate runs, for scene runs without a scene_tests stage')
    p.add_argument('--output',type=Path,default=None);a=p.parse_args()
    root=a.root.resolve();out=(a.output or root/'evidence/obstacle_scenes').resolve();out.mkdir(parents=True,exist_ok=True)
    suites=read(root/'runtime/tests/suite_manifest.json')['suites']
    pair_type={}
    for item in suites:
        with np.load(root/'runtime/tests/suites'/item['file'],allow_pickle=True) as z:pair_type[item['task']]=z['pair_type'].astype(int)
    runs={'no_wall':a.no_wall.resolve()}
    for run in a.runs:
        status=read(run/'status.json');runs[Path(status['scene']).stem]=run.resolve()
    scenes=[name for name in runs if name!='no_wall']
    for name,run in runs.items():
        status=read(run/'status.json');assert status['complete'] and not status['smoke'],name
    audits={name:read(runs[name]/'audit/endpoint_audit.json') for name in scenes}
    # The shared audit calls every added obstacle a wall.
    valid={name:{t:{r['query_id']:r['wall_endpoints_valid'] for r in v['per_query']} for t,v in audit['tasks'].items()} for name,audit in audits.items()}
    summary=dict(runs={k:str(v) for k,v in runs.items()},scenes={},results={},paired={});per={}
    for name,audit in audits.items():
        blocked=Counter(TYPES[pair_type[t][r['query_id']]] for t,v in audit['tasks'].items() for r in v['per_query'] if not r['wall_endpoints_valid'])
        summary['scenes'][name]=dict(obstacles=audit['walls'],valid_endpoints=audit['wall_valid'],blocked_by_type=dict(blocked))
    for name,run in runs.items():
        status=read(run/'status.json');res=dict(status=status,methods={})
        for method,prefix,n in [('mppi','bellman_mppi_',1800),('rk4','bellman_rk4_',600)]:
            rows=flags(run,prefix);assert sum(len(v) for v in rows.values())==n,(name,method);per.setdefault(name,{})[method]=rows
            ok=lambda subset=None,kind=None:sum(v['success'] for l,qs in rows.items() for q,v in qs.items()
                if (subset is None or valid[subset][task_of(l)][int(q)]) and (kind is None or TYPES[pair_type[task_of(l)][int(q)]]==kind))
            count=lambda subset,kind=None:sum(valid[subset][task_of(l)][int(q)] for l,qs in rows.items() for q in qs
                if kind is None or TYPES[pair_type[task_of(l)][int(q)]]==kind)
            by_type=Counter();by_type_n=Counter();by_task=Counter();reasons=Counter()
            for l,qs in rows.items():
                for q,v in qs.items():
                    kind=TYPES[pair_type[task_of(l)][int(q)]];by_type_n[kind]+=1;by_type[kind]+=bool(v['success']);by_task[task_of(l)]+=bool(v['success'])
                    if not v['success']:reasons.update(v['reasons'])
            res['methods'][method]=dict(n=n,success=ok(),
                subsets={s:dict(n=count(s),success=ok(s),by_type={k:dict(n=count(s,k),success=ok(s,k)) for k in TYPES}) for s in valid},
                by_type={k:dict(success=by_type[k],n=by_type_n[k]) for k in TYPES},by_task=dict(by_task),failure_reasons=dict(reasons))
        prepared=run/'prepared'
        if (prepared/'bank/manifest.json').exists():
            bank=read(prepared/'bank/manifest.json');reports=bank['rollforward']['reports']
            res['bank']=dict(anchors=bank['anchors'],safe_edge_fraction=float(np.mean([r['safe_edge_fraction'] for r in reports])))
        if (prepared/'speed_audit.json').exists():res['speed_validation']=read(prepared/'speed_audit.json')['validation']
        if (run/'sampled/complete.json').exists():
            c=read(run/'sampled/complete.json');res['sampling']=dict(pool_size_each=c['pool_size_each'],original_pairs=c['original_pairs'],augmentation_pairs=c['augmentation_pairs'])
        summary['results'][name]=res
    # Same query, same planning seed, with and without the obstacles; endpoint-valid queries only.
    for s in scenes:
        for m in ('mppi','rk4'):
            pairs=Counter();ours=Counter();theirs=Counter()
            for l,qs in per[s][m].items():
                for q,v in qs.items():
                    if not valid[s][task_of(l)][int(q)]:continue
                    w=per['no_wall'][m][l][q];pairs[(bool(w['success']),bool(v['success']))]+=1
                    if not v['success']:ours.update(v['reasons'])
                    if not w['success']:theirs.update(w['reasons'])
            summary['paired'].setdefault(s,{})[m]=dict(both_succeed=pairs[(True,True)],lost=pairs[(True,False)],gained=pairs[(False,True)],
                both_fail=pairs[(False,False)],failure_reasons=dict(ours),no_wall_failure_reasons=dict(theirs))
    # Scene test suites: the pipeline's scene_tests stage, or a separately registered obstacle-evaluate run of the same checkpoint.
    folders={name:runs[name]/'scene_tests' for name in scenes if (runs[name]/'scene_tests/evaluation.json').exists()}
    for d in a.tests_eval:
        ev=read(d/'evaluation/evaluation.json');name=Path(ev['scene']).stem;assert ev['complete'] and not ev['smoke'],d
        assert hashlib.sha256((runs[name]/'training/checkpoint.pt').read_bytes()).hexdigest()==ev['checkpoint_sha256'],d;folders[name]=d.resolve()/'evaluation'
    summary['scene_tests']={}
    for name in [s for s in scenes if s in folders]:
        folder=folders[name];ev=read(folder/'evaluation.json');tests=root/'experiments/obstacle_scenes/tests'/name;rows=[]
        for item in read(tests/'suite_manifest.json')['suites']:
            with np.load(tests/'suites'/item['file'],allow_pickle=True) as z:direct,kind=int(z['direct_line_strict_feasible'].sum()),TYPES[int(z['pair_type'][0])]
            got=lambda m:sum(r['success'] for r in ev['scores'] if r['task']==item['task'] and r['method']==m)
            rows.append(dict(suite=item['task'],pair_type=kind,n=item['n'],direct_line=direct,rk4=got('native_rk4'),mppi=got('matched_mppi')))
        summary['scene_tests'][name]=dict(evaluation=str(folder),mppi_successes=ev['mppi_successes'],native_success=ev['native_success'],suites=rows)
    # Reference: the no-wall run on the frozen certified mixed suite, built the same way as the general suites.
    mixed=pair_type['mixed'];summary['reference_mixed']={}
    for method in ('mppi','rk4'):
        rows=per['no_wall'][method];summary['reference_mixed'][method]={k:dict(n=sum(1 for l,qs in rows.items() if task_of(l)=='mixed' for q in qs if TYPES[mixed[int(q)]]==k),
            success=sum(v['success'] for l,qs in rows.items() if task_of(l)=='mixed' for q,v in qs.items() if TYPES[mixed[int(q)]]==k)) for k in TYPES}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')

    R=summary['results'];S=summary['scenes']
    def pct(k,n):return f'{k}/{n} ({100*k/n:.2f}%)' if n else '—'
    def label(name):return name.replace('_','-')
    L=['# Obstacle scenes: from-source retraining','',
       'Every run used the same from-source pipeline, seeds and machine (RTX 3090, derived sm_86 image), so they differ only in the scene.',
       'Scene files: `experiments/obstacle_scenes/scenes/`.','',
       '## Scenes','','|Scene|Valid endpoints|Blocked: cupping / pouring / cross|','|---|---:|---|']
    for name,s in S.items():
        b=s['blocked_by_type'];L.append(f"|{name}|{s['valid_endpoints']}/600|{b.get('cupping',0)} / {b.get('pouring',0)} / {b.get('cross',0)}|")
    L+=['','|Scene|Box|x (m)|y (m)|z (m)|','|---|---|---|---|---|']
    for name,s in S.items():
        for box in s['obstacles']:
            (px,py,pz),(dx,dy,dz)=box['pose'][:3],box['dims']
            L.append(f"|{name}|{box['name']}|{px-dx/2:.2f}–{px+dx/2:.2f}|{py-dy/2:+.2f}–{py+dy/2:+.2f}|{pz-dz/2:.3f}–{pz+dz/2:.3f}|")
    L+=['','## Success rates (all 600 queries)','','|Run|MPPI seeds 0/1/2|MPPI|RK4|','|---|---|---:|---:|']
    for name,r in R.items():
        L.append(f"|{label(name)} ({Path(summary['runs'][name]).name})|{r['status']['mppi_successes']}|{pct(r['methods']['mppi']['success'],1800)}|{pct(r['methods']['rk4']['success'],600)}|")
    if summary['scene_tests']:
        L+=['','## Scene test suites','','Each scene\'s own frozen, certified queries (`experiments/obstacle_scenes/tests/<scene>/`): every query has a witness path that the release scorer accepts in the scene, so no query is impossible. Direct line: queries whose straight joint-space line already passes.','',
            '|Scene|Suite|Pair type|Direct line|RK4|MPPI, 3 seeds|','|---|---|---|---:|---:|---:|']
        for name,x in summary['scene_tests'].items():
            for r in x['suites']:L.append(f"|{name}|{r['suite']}|{r['pair_type']}|{r['direct_line']}/{r['n']}|{pct(r['rk4'],r['n'])}|{pct(r['mppi'],3*r['n'])}|")
            n=sum(r['n'] for r in x['suites']);L.append(f"|**{name}**|**all**||{sum(r['direct_line'] for r in x['suites'])}/{n}|**{pct(x['native_success'],n)}**|**{pct(sum(x['mppi_successes']),3*n)}**|")
        ref=summary['reference_mixed']
        L+=['','For reference, the no-wall run on the frozen certified `mixed` suite (same construction, no obstacles; 30 cupping, 30 pouring, 40 cross queries): '
            +', '.join(f"{k} MPPI {pct(ref['mppi'][k]['success'],ref['mppi'][k]['n'])}, RK4 {pct(ref['rk4'][k]['success'],ref['rk4'][k]['n'])}" for k in TYPES)
            +'. Certified random cross pairs are much harder than the cross segments of the five task suites, so these totals are not comparable with the frozen 600.']
    L+=['','## Same queries with and without the obstacles','',
        'Restricted to the queries whose endpoints are valid in each scene, so impossible queries do not count against the scene runs.','',
        '|Subset|Planner|No wall|With obstacles|','|---|---|---:|---:|']
    for s in scenes:
        for m in ('mppi','rk4'):
            nw=R['no_wall']['methods'][m]['subsets'][s];ww=R[s]['methods'][m]['subsets'][s]
            L.append(f"|{s} valid ({S[s]['valid_endpoints']} queries)|{m.upper()}|{pct(nw['success'],nw['n'])}|{pct(ww['success'],ww['n'])}|")
    L+=['','## MPPI by query type, endpoint-valid queries','','|Scene|Query type|No wall|With obstacles|','|---|---|---:|---:|']
    for s in scenes:
        for k in TYPES:
            nw=R['no_wall']['methods']['mppi']['subsets'][s]['by_type'][k];ww=R[s]['methods']['mppi']['subsets'][s]['by_type'][k]
            L.append(f'|{s}|{k}|{pct(nw["success"],nw["n"])}|{pct(ww["success"],ww["n"])}|')
    L+=['','## Paired outcomes, endpoint-valid queries','','Same query and planning seed with and without the obstacles. Lost: solved without them, failed with them.','',
        '|Scene|Planner|Both succeed|Lost|Gained|Both fail|Failure reasons with obstacles|Failure reasons without|','|---|---|---:|---:|---:|---:|---|---|']
    reasons=lambda c:', '.join(f'{k} {v}' for k,v in sorted(c.items(),key=lambda x:-x[1])) or '—'
    for s in scenes:
        for m in ('mppi','rk4'):
            x=summary['paired'][s][m]
            L.append(f"|{s}|{m.upper()}|{x['both_succeed']}|{x['lost']}|{x['gained']}|{x['both_fail']}|{reasons(x['failure_reasons'])}|{reasons(x['no_wall_failure_reasons'])}|")
    L+=['','## MPPI by task (/300, all queries)','','|Task|'+'|'.join(map(label,R))+'|','|---|'+'---:|'*len(R)]
    for t in R['no_wall']['methods']['mppi']['by_task']:
        L.append(f'|{t}|'+'|'.join(str(r['methods']['mppi']['by_task'][t]) for r in R.values())+'|')
    L+=['','## Failure reasons (all queries, including impossible ones; a run can have several)','','|Run|MPPI|RK4|','|---|---|---|']
    for name,r in R.items():
        f=lambda m:', '.join(f'{k} {v}' for k,v in sorted(r['methods'][m]['failure_reasons'].items(),key=lambda x:-x[1]))
        L.append(f"|{label(name)}|{f('mppi')}|{f('rk4')}|")
    L+=['','## Data, training and timing','','|Run|Bank rows|Bank safe-move fraction|Speed log-RMSE (orig / aug)|Stage seconds|Total|','|---|---:|---:|---|---|---:|']
    for name,r in R.items():
        b=r.get('bank',{});speed=' / '.join('%.3f'%v['log_rmse'] for v in r.get('speed_validation',[]))
        L.append(f"|{label(name)}|{b.get('anchors','—')}|{b.get('safe_edge_fraction',float('nan')):.3f}|{speed}|"
                 f"{', '.join(f'{k} {v:.0f}' for k,v in r['status']['timing'].items())}|{r['status']['elapsed_seconds']/60:.1f} min|")
    L+=['','## Caveats','',
        '- Modified environment (cuRobo rebuilt for sm_86, host CUDA driver libraries): independent samples, not bitwise replays of the original machine.',
        '- One training seed per scene; the 600 queries are a reused development set. No significance or generalization claims.',
        '- T(q,g) takes only q and g. The obstacles reach it only through the scene-specific training data, speed labels and bank.',
        '- In the frozen 600, valid endpoints do not prove a collision-free path exists; every scene-suite query has a witness path.',
        '- The scene suites keep only queries the certifying planner solved, so they lean towards queries it finds easier.',
        '- The 600 queries were drawn for the empty table, not for these obstacles: they test planning around the obstacles, not starts and goals placed on or inside them.',
        '- The frozen no-wall baseline and committed evidence are unchanged.']
    (out/'COMPARISON.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))

if __name__=='__main__':main()
