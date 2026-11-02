"""Read-only: for every saved Task 2 path of the P-NTFields planner and of both fields under ntrl-demo's MPPI (seed 0),
the first contact along the path (cabinet or self, as a fraction of the path) and the joint-space detour (path length /
straight distance, max-norm), with the release sphere model at the scorer's margins (as the viewer export).

  ~/.venvs/bellman-ntfield/bin/python experiments/pntfields_cabinet/analysis/contact_analysis.py contacts.json
"""
import json,sys,collections
from pathlib import Path
import numpy as np, torch
V=Path(__file__).resolve().parents[1]/'visualizer'; sys.path.insert(0,str(V))
import export_cabinet as X
torch.set_num_threads(8)
ROOT=X.ROOT; out=collections.defaultdict(list)
for scene,(tests,*runs) in X.SCENES.items():
    pc=X.PieceClearance(ROOT/'experiments/cabinet/scenes'/f'{scene}.yml','split','cpu')
    paths={k:ROOT/'results'/f'registered_{v}'/'evaluation' for k,v in zip(['pnt','ntrl'],runs)}
    man=json.loads((ROOT/'experiments/cabinet'/tests/'suite_manifest.json').read_text())
    for item in man['suites']:
        t=item['task']; z=np.load(ROOT/'experiments/cabinet'/tests/'suites'/item['file'],allow_pickle=True); line=z['direct_line_strict_feasible']
        for src in ('pnt_plan','pnt_ntrl','ntrl_ntrl'):
            run,label=X.LABEL[src]; label=label.format(t=t)
            with np.load(paths[run]/(label+'.npz'),allow_pickle=True) as zz: traj=list(zz['trajectories']); ids=[int(x) for x in zz['query_ids']] if 'query_ids' in zz.files else list(range(len(traj)))
            score=json.loads((paths[run]/(label+'.score.json')).read_text())[label]['per_query']
            for k,qi in enumerate(ids):
                p=np.asarray(traj[k],np.float64); s=score[str(qi)]
                L=np.abs(np.diff(p,axis=0)).max(1).sum() if len(p)>1 else 0.; D=np.abs(p[-1]-p[0]).max()
                rec=dict(scene=scene,suite=t,src=src,q=qi,ok=bool(s['success']),line=bool(line[qi]),detour=float(L/max(D,1e-6)))
                if 'collision' in s['reasons']:
                    d,ends=X.dense(p,X.STEP)
                    with torch.no_grad():
                        qn=(torch.tensor(d,dtype=torch.float32)-pc.robot.q_center)/pc.robot.q_range
                        pcs=torch.cat([pc.pieces(qn[i:i+4096]) for i in range(0,len(qn),4096)]).numpy()
                    env,slf=pcs[:,0],pcs[:,1]/X.SELF_SCALE; bad_e=env<X.OBS_MARGIN; bad_s=slf<X.SELF_MARGIN
                    fe=np.argmax(bad_e) if bad_e.any() else None; fs=np.argmax(bad_s) if bad_s.any() else None
                    first=min([x for x in (fe,fs) if x is not None])
                    rec.update(kind='env' if fe is not None and fe==first else 'self',frac=float(first/max(len(d)-1,1)),min_env=float(env.min()),env_any=fe is not None,self_any=fs is not None)
                out[src].append(rec)
    print(scene,'done',flush=True)
json.dump(out,open(sys.argv[1],'w'))
