"""Compare the Bellman and metric NTFields cabinet runs on the frozen suites of one cabinet scene.

    python3 experiments/cabinet/report.py [--scene cabinet_plus10] --bellman results/registered_<cabinet run> \
        [--bellman-eval results/registered_<cabinet-evaluate run>] --metric results/registered_<cabinet-metric run> \
        [--metric-eval results/registered_<cabinet-evaluate run>]

Training inputs come from the pipeline runs; scores from the evaluation runs (a pipeline's own
evaluation/ when no separate evaluation run is given). The checkpoint hashes must match.

Writes COMPARISON.md and comparison.json into evidence/cabinet/ (scene cabinet) or
evidence/cabinet/<variant>/ (scene cabinet_<variant>). Standard library only.
"""
import argparse,datetime as dt,hashlib,json,os,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1].parent
REGISTRY=Path(os.environ.get('NTFIELDS_REGISTRY','~/.local/share/ntfields-experiments')).expanduser()
PLANNERS=[('bellman','native_rk4','Bellman, RK4 (native)'),('bellman','matched_mppi','Bellman, release MPPI'),
          ('metric','native_rk4','Metric NTFields, RK4'),('metric','matched_mppi','Metric NTFields, release MPPI'),
          ('metric','ntrl_mppi','Metric NTFields, ntrl-demo MPPI (native)')]


def load(run,evaluation_run=None):
    run=Path(run).resolve();status=json.loads((run/'status.json').read_text());progress=json.loads((run/'training/progress.json').read_text())
    assert not status['smoke'] and progress['complete'],f'{run.name} has no complete full training'
    source=Path(evaluation_run).resolve() if evaluation_run else run
    evaluation=json.loads((source/'evaluation/evaluation.json').read_text())
    assert evaluation['complete'] and not evaluation['smoke'],f'{source.name}: evaluation is not complete'
    assert evaluation['checkpoint_sha256']==sha(run/'training/checkpoint.pt'),f'{source.name} scored another checkpoint'
    return run,status,evaluation,source


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(s,n):return f'{s}/{n} ({100*s/n:.1f}%)'


def clearance_scale(scene):
    """The scene's `variant: clearance_scale:` (make_scenes.py); 1 for the cabinet scene."""
    m=re.search(r'^  clearance_scale: ([0-9.]+)$',(ROOT/'experiments/cabinet/scenes'/(scene+'.yml')).read_text(),re.M)
    return float(m.group(1)) if m else 1.


def record(run):
    """The registry record of a registered run directory (results/registered_<id>), or None."""
    path=REGISTRY/'runs'/run.name.replace('registered_','')/'record.json'
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--bellman',type=Path,required=True);p.add_argument('--metric',type=Path,required=True)
    p.add_argument('--bellman-eval',type=Path);p.add_argument('--metric-eval',type=Path);p.add_argument('--scene',default='cabinet');a=p.parse_args()
    runs={k:load(v,e) for k,v,e in [('bellman',a.bellman,a.bellman_eval),('metric',a.metric,a.metric_eval)]}
    ev={k:v[2] for k,v in runs.items()}
    assert ev['bellman']['method']=='bellman' and ev['metric']['method']=='metric'
    assert ev['bellman']['tests_manifest_sha256']==ev['metric']['tests_manifest_sha256'],'runs scored different suites'
    assert ev['bellman']['scene_sha256']==ev['metric']['scene_sha256'] and all(Path(e['scene']).stem==a.scene for e in ev.values()),f'runs are not in scene {a.scene}'
    suffix=a.scene[len('cabinet'):];tests=ROOT/'experiments/cabinet'/('tests'+suffix);manifest=json.loads((tests/'suite_manifest.json').read_text())
    assert sha(tests/'suite_manifest.json')==ev['bellman']['tests_manifest_sha256'],'scored suites are not the frozen suites of this scene'
    suites=[s['task'] for s in manifest['suites']];n=manifest['suites'][0]['n']
    rows={}
    for method,planner,label in PLANNERS:
        scores=[r for r in ev[method]['scores'] if r['method']==planner]
        rows[(method,planner)]=dict(label=label,total=sum(r['success'] for r in scores),n=n*len(scores),
            per_suite={s:(sum(r['success'] for r in scores if r['task']==s),n*sum(1 for r in scores if r['task']==s)) for s in suites},
            per_seed=sorted({r['planning_seed'] for r in scores if r['planning_seed'] is not None}),
            failures={k:sum(r['failures'].get(k,0) for r in scores) for k in sorted({k for r in scores for k in r['failures']})})
    # Paired outcomes on identical queries, same planner and planning seed.
    paired={}
    for planner in ['native_rk4','matched_mppi']:
        both=only_b=only_m=neither=0
        for s in suites:
            for seed in ([None] if planner=='native_rk4' else [0,1,2]):
                rb=[r for r in ev['bellman']['scores'] if r['method']==planner and r['task']==s and r['planning_seed']==seed][0]
                rm=[r for r in ev['metric']['scores'] if r['method']==planner and r['task']==s and r['planning_seed']==seed][0]
                b,m=set(rb['success_ids']),set(rm['success_ids'])
                both+=len(b&m);only_b+=len(b-m);only_m+=len(m-b);neither+=n-len(b|m)
        paired[planner]=dict(both=both,bellman_only=only_b,metric_only=only_m,neither=neither)
    out=ROOT/'evidence/cabinet'/suffix.lstrip('_');out.mkdir(parents=True,exist_ok=True)
    generation=json.loads((tests/'generation.json').read_text());frozen=json.loads((tests/'FROZEN.json').read_text())
    data={k:runs[k][1] for k in runs};timing={}
    for k in runs:
        timing[k]={stage:sec for stage,sec in data[k]['timing'].items()}
        if runs[k][3]!=runs[k][0]:
            # Separate evaluation run: wall time from its registry record.
            r=record(runs[k][3])
            if r:timing[k]['evaluate']=(dt.datetime.fromisoformat(r['ended'])-dt.datetime.fromisoformat(r['started'])).total_seconds()
    gpus={k:sorted({r['spec']['config']['gpus'] for r in map(record,{runs[k][0],runs[k][3]}) if r}) for k in runs}
    doc=dict(scene=a.scene,bellman_run=runs['bellman'][0].name,metric_run=runs['metric'][0].name,bellman_evaluation=runs['bellman'][3].name,metric_evaluation=runs['metric'][3].name,tests=frozen,suites=suites,queries_per_suite=n,
             results={f'{m}/{p}':v for (m,p),v in rows.items()},paired=paired,timing=timing,gpus=gpus)
    (out/'comparison.json').write_text(json.dumps(doc,indent=2)+'\n')
    B,M=runs['bellman'][0],runs['metric'][0]
    prepared=json.loads((B/'prepared/complete.json').read_text());speed=json.loads((B/'prepared/speed_audit.json').read_text())
    sampled=json.loads((B/'sampled/complete.json').read_text());mdata=json.loads((M/'data/complete.json').read_text());mtrain=json.loads((M/'training/progress.json').read_text())
    title=f' ({a.scene}: {round(100*(clearance_scale(a.scene)-1))}% more clearance)' if suffix else ''
    L=['# Cabinet task'+title+': Bellman NTFields vs metric NTFields','',
       f'Training runs: Bellman `{B.name}`, metric NTFields `{M.name}`; '+(f'evaluation runs `{runs["bellman"][3].name}` and `{runs["metric"][3].name}`. '
       if any(runs[k][3]!=runs[k][0] for k in runs) else 'scored by their own evaluation stage. ')+
       'Both scored on the same frozen suites of this scene '
       f'(`{tests.relative_to(ROOT)}/`, from `{frozen["source_run"]}`): {len(suites)} suites x {n} certified queries, '
       'release scorer in the cabinet scene with the tilt gate off. MPPI rows count 3 planning seeds per query.','',
       '## Success rate, all 600 queries','','|Method and planner|Success|','|---|---:|']
    for (m,p),v in rows.items():L.append(f"|{v['label']}|{pct(v['total'],v['n'])}|")
    L+=['','## Per suite','','|Suite|'+'|'.join(v['label'] for v in rows.values())+'|','|---|'+'---:|'*len(rows)]
    for s in suites:L.append(f'|{s}|'+'|'.join(pct(*v['per_suite'][s]) for v in rows.values())+'|')
    L+=['','## Same planner, same queries','','|Planner|Both solve|Bellman only|Metric only|Neither|','|---|---:|---:|---:|---:|']
    for planner,label in [('native_rk4','RK4 (deterministic, 600)'),('matched_mppi','Release MPPI (600 x 3 seeds)')]:
        q=paired[planner];L.append(f"|{label}|{q['both']}|{q['bellman_only']}|{q['metric_only']}|{q['neither']}|")
    L+=['','## Failure reasons (scorer)','','|Method and planner|'+'|'.join(['collision','endpoint','joint_limits'])+'|','|---|---:|---:|---:|']
    for v in rows.values():L.append(f"|{v['label']}|"+'|'.join(str(v['failures'].get(k,0)) for k in ['collision','endpoint','joint_limits'])+'|')
    L+=['','`endpoint`: the path does not end at the goal (planner did not converge and its last segment was not appended or failed).','',
        '## Training inputs','',
        f"- Bellman: FK pool {sampled['pool']['size']:,} configurations in the sampling box ({100*sampled['pool']['cupboard_share']:.2f}% with the grasp point in the cupboard), "
        f"1,000,000 original + {sampled['augmentation_pairs']:,} coverage pairs; speed network held-out MAE "
        +', '.join(f"{x['source']} {x['mae']:.4f}" for x in speed['validation'])+f"; bank {prepared['bank_anchors']:,} anchors.",
        f"- Metric NTFields: {mdata['pairs']:,} pairs ({100*mdata['x0_cupboard_share']:.2f}% of x0 in the cupboard), {mtrain['epochs']} epochs, "
        f"final-epoch loss {float(mtrain['final_checkpoint'].rsplit('ValLoss_',1)[1][:-3]):.3g} (ntrl-demo loss_n).",
        '','## Wall time (seconds; RTX 3090, GPU of each registered run in brackets)','']
    for k,t in timing.items():L.append(f'- {k} [{", ".join(gpus[k])}]: '+', '.join(f'{stage} {sec:.0f}' for stage,sec in t.items())+f'; total {sum(t.values()):.0f}')
    diagnostics=out/'diagnostics.json'
    if diagnostics.is_file():
        d=json.loads(diagnostics.read_text())
        if d['tests_manifest_sha256']==ev['bellman']['tests_manifest_sha256']:
            sl=d['straight_line'];L+=['',f'## Diagnostics (`diagnose.py`, read-only; `{diagnostics.relative_to(ROOT)}`)','',
                f"Straight joint-space line (the suites' own certificates): {sl['total']}/600 ({100*sl['total']/600:.1f}%); per suite "
                +', '.join(f'{t} {sl[t]}' for t in suites)+'.','',
                'First scorer violation on every path rejected for collision (0.001 rad densification):','',
                '|Method and planner|Collision paths|Cupboard panel|Table|Self|Grasp point at first contact: free box / compartment / other cupboard / outside box|','|---|---:|---:|---:|---:|---|']
            for key,c in d['collisions'].items():
                r=c['region'];comp=sum(v for k,v in r.items() if k.startswith('compartment'))
                L.append(f"|{key}|{c['n']}|{c['culprit_group']['cupboard_panel']}|{c['culprit_group']['table']}|{c['culprit_group']['self']}|"
                         f"{r.get('box',0)} / {comp} / {r.get('cupboard_other',0)} / {r.get('outside_box',0)}|")
            cov=d['coverage'];L+=['','Training states with the grasp point in the cupboard (outer box) and in a compartment interior:','','|States|Cupboard|Compartments|','|---|---:|---:|']
            for key,c in cov.items():L.append(f"|{key}|{100*c['cupboard']:.2f}%|{100*sum(v for k,v in c.items() if k.startswith('compartment_')):.2f}%|")
            L+=['','Implied speed 1/|grad_q T(q, goal)| along the certified witness paths (points more than 0.02 from the goal), against the true clearance speed:','',
                '|Suite|True, in cupboard|Bellman|Metric|True, outside|Bellman|Metric|Log-ratio RMSE in cupboard: Bellman / metric|','|---|---:|---:|---:|---:|---:|---:|---|']
            f=lambda v:'—' if v is None else f'{v:.2f}'
            for t in suites:
                r=d['implied_speed'][t]
                L.append(f"|{t}|{f(r.get('true_speed_mean_inside'))}|{f(r.get('bellman_implied_mean_inside'))}|{f(r.get('metric_implied_mean_inside'))}|"
                         f"{f(r.get('true_speed_mean_outside'))}|{f(r.get('bellman_implied_mean_outside'))}|{f(r.get('metric_implied_mean_outside'))}|"
                         f"{f(r.get('bellman_log_ratio_rmse_inside'))} / {f(r.get('metric_log_ratio_rmse_inside'))}|")
    L+=['','One training seed per method; planning seeds are not training seeds. No significance claim.','']
    (out/'COMPARISON.md').write_text('\n'.join(L));print('\n'.join(L))


if __name__=='__main__':main()
