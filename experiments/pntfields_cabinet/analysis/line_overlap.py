"""Read-only: per execution, how each method's scorer verdicts overlap the straight joint-space line (the suites'
direct_line_strict_feasible flag) and each other, per suite and per scene. Run from the repository root.

  ~/.venvs/bellman-ntfield/bin/python experiments/pntfields_cabinet/analysis/line_overlap.py
"""
import json,numpy as np
from pathlib import Path
R=Path('results');C=Path('experiments/cabinet')
S={'cabinet':('tests','20261026T080627Z-53c1dd59','20261026T041234Z-664b5d50'),
 'cabinet_pi':('tests_pi','20261026T110153Z-44e64765','20261026T041504Z-379bcde5'),
 'cabinet_w25':('tests_w25','20261026T140052Z-b8746f48','20261026T022529Z-b25f368e'),
 'cabinet_w20':('tests_w20','20261026T165453Z-9a9b9b4d','20261026T022740Z-c32709b5'),
 'cabinet_w15':('tests_w15','20261026T194913Z-656fe377','20261026T023140Z-b00193b9')}
def sc(run,label):
    p=R/f'registered_{run}'/'evaluation'/f'{label}.score.json'
    d=json.loads(p.read_text())[label]['per_query'];return np.array([d[str(i)]['success'] for i in range(len(d))])
tot={};scene_tot={}
for s,(t,P,N) in S.items():
    man=json.loads((C/t/'suite_manifest.json').read_text())
    for it in man['suites']:
        task=it['task'];line=np.load(C/t/'suites'/it['file'],allow_pickle=True)['direct_line_strict_feasible'].astype(bool)
        rows={'line':np.tile(line,3),
              'pnt_ntrl':np.concatenate([sc(P,f'pnt_ntrl_mppi_{task}_seed{k}') for k in range(3)]),
              'ntrl_ntrl':np.concatenate([sc(N,f'metric_ntrl_mppi_{task}_seed{k}') for k in range(3)]),
              'pnt_plan':np.tile(sc(P,f'pnt_plan_{task}'),3),'pnt_rk4':np.tile(sc(P,f'pnt_rk4_{task}'),3),'ntrl_rk4':np.tile(sc(N,f'metric_rk4_{task}'),3)}
        L=rows['line']
        for k,v in rows.items():
            for key in ((task,k),(s,k)):
                a=tot.setdefault(key,[0,0,0,0]);a[0]+=v.sum();a[1]+=(v&~L).sum();a[2]+=(~v&L).sum();a[3]+=len(v)
        p,n=rows['pnt_ntrl'],rows['ntrl_ntrl']
        for key in ((task,'PvN'),(s,'PvN')):
            a=tot.setdefault(key,[0,0,0,0]);a[0]+=(p&n).sum();a[1]+=(p&~n).sum();a[2]+=(~p&n).sum();a[3]+=len(p)
tasks=['free_to_free','free_to_compartment','compartment_to_free','compartment_to_compartment','same_compartment','anywhere']
print('x3 executions: (success, wins where line fails, losses where line ok); PvN = (both, P only, N only)')
for grp in (tasks,list(S)):
    for k in ['line','pnt_ntrl','ntrl_ntrl','pnt_plan','pnt_rk4','ntrl_rk4','PvN']:
        print(k.ljust(10),[tuple(int(x) for x in tot[(t,k)][:3]) for t in grp], 'sum',[int(sum(tot[(t,k)][j] for t in grp)) for j in range(4)])
    print()
