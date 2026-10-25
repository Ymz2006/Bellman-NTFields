"""Clearance sweep of the cabinet task: Bellman NTFields vs metric NTFields in the cabinet scene and in
its 10% and 20% larger-clearance variants (make_scenes.py).

    python3 experiments/cabinet/clearance_report.py

Reads, per scene, the comparison and diagnostics that report.py and diagnose.py wrote
(evidence/cabinet/, evidence/cabinet/plus10/, evidence/cabinet/plus20/), the frozen suites' generation
records and the pipeline runs' training-input summaries. Writes evidence/cabinet/CLEARANCE.md and
clearance.json. Each scene is scored on its own frozen, certified suites, so the scenes share no
queries; the straight joint-space line on each scene's suites is the reference for how hard they are.
"""
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT=ROOT/'experiments/cabinet'
SCENES=['cabinet','cabinet_plus10','cabinet_plus20']
ROWS=[('bellman/native_rk4','Bellman, RK4 (native)'),('bellman/matched_mppi','Bellman, release MPPI'),('metric/native_rk4','Metric NTFields, RK4 (native)'),
      ('metric/matched_mppi','Metric NTFields, release MPPI'),('metric/ntrl_mppi','Metric NTFields, ntrl-demo MPPI')]
COMPARTMENT_SUITES=['free_to_compartment','compartment_to_free','compartment_to_compartment','same_compartment']
OPEN_SUITES=['free_to_free','anywhere']
# The reading below was written for these runs (scene: Bellman run, metric NTFields run); it is left out
# when the report is regenerated from other runs.
READING_RUNS={'cabinet':('registered_20261019T034213Z-f9c501f9','registered_20261019T034237Z-30d4d872'),
              'cabinet_plus10':('registered_20261019T171925Z-c5ebfc5b','registered_20261019T172013Z-4aa0e26a'),
              'cabinet_plus20':('registered_20261019T171949Z-c41ae437','registered_20261019T172037Z-75a8efe5')}
READING=['## Reading','',
 '- The geometry is roomier, as intended: the true clearance speed at the compartment endpoints rises from 0.33 to 0.37 and 0.41, '
 'and along the certified witness paths inside the cupboard from 0.27 to 0.29 and 0.32.',
 '- The queries did not get easier: the straight line solves 31-35% of each scene\'s queries and 10-12% of its compartment queries. '
 'More room also makes more of every compartment reachable (test-suite compartment pools 1.8x and 3.0x larger), and the compartment '
 'endpoints lie slightly deeper (0.086 -> 0.093 -> 0.098 m past the opening).',
 '- Neither method turns the extra room into success. Overall rates move by at most 2.3 points, about one standard error of the '
 'query draw. On the compartment suites Bellman drifts down (release MPPI 13.0 -> 12.1 -> 10.3%) and metric NTFields up '
 '(9.6 -> 9.8 -> 10.8%), so Bellman\'s lead with the same planner shrinks from +5.2 to +4.6 and +1.7 points overall. Bellman still '
 'solves more queries alone than metric NTFields in every scene (RK4: 65 vs 18, 61 vs 20, 50 vs 21).',
 '- What limits both methods is unchanged by the clearance: PANEL_SHARE of the collision failures hit a cupboard panel, mostly early in '
 'the path, and inside the cupboard both travel-time fields imply about 2x (Bellman) and 3x (metric NTFields) the true clearance '
 'speed. The extra room raises that true speed by at most 20%, much less than these errors.',
 '- One training seed per method and scene: the drift of Bellman\'s compartment rate is within what training-seed and query-draw '
 'variation could produce; separating them needs repeated training seeds.','']


def load(path):return json.loads(Path(path).read_text())


def pct(s,n):return f'{s}/{n} ({100*s/n:.1f}%)'


def rate(s,n):return f'{100*s/n:.1f}%'


def scene_info(scene):
    suffix=scene[len('cabinet'):];doc=yaml.safe_load((EXPERIMENT/'scenes'/(scene+'.yml')).read_text())
    panels=[c for c in doc['cuboids'] if c['name'].startswith('shelf_')]
    lo=[min(c['pose'][k]-c['dims'][k]/2 for c in panels) for k in range(3)];hi=[max(c['pose'][k]+c['dims'][k]/2 for c in panels) for k in range(3)]
    comp={c['name']:c for c in doc['compartments']};top=[c for c in panels if c['name']=='shelf_top'][0]
    evidence=ROOT/'evidence/cabinet'/suffix.lstrip('_');tests=EXPERIMENT/('tests'+suffix)
    comparison=load(evidence/'comparison.json');diagnostics=load(evidence/'diagnostics.json') if (evidence/'diagnostics.json').is_file() else None
    assert comparison.get('scene',scene)==scene and comparison['tests']['suite_manifest_sha256']==load(tests/'FROZEN.json')['suite_manifest_sha256']
    if diagnostics:assert diagnostics['tests_manifest_sha256']==comparison['tests']['suite_manifest_sha256'],f'{scene}: diagnostics of other suites'
    bellman,metric=ROOT/'results'/comparison['bellman_run'],ROOT/'results'/comparison['metric_run']
    return dict(scene=scene,scale=(doc.get('variant') or {}).get('clearance_scale',1.),evidence=str(evidence.relative_to(ROOT)),tests=str(tests.relative_to(ROOT)),
        opening_width=round(top['dims'][1],4),heights={k:round(v['z'][1]-v['z'][0],4) for k,v in comp.items()},
        cupboard=dict(depth=round(hi[0]-lo[0],4),width=round(hi[1]-lo[1],4),height=round(hi[2]-lo[2],4)),sampling=doc['sampling'],
        comparison=comparison,diagnostics=diagnostics,generation=load(tests/'generation.json'),frozen=load(tests/'FROZEN.json'),
        sampled=load(bellman/'sampled/complete.json'),prepared=load(bellman/'prepared/complete.json'),speed=load(bellman/'prepared/speed_audit.json'),
        metric_data=load(metric/'data/complete.json'),metric_training=load(metric/'training/progress.json'))


def group(result,suites):
    s=sum(result['per_suite'][x][0] for x in suites);n=sum(result['per_suite'][x][1] for x in suites);return s,n


def main():
    info=[scene_info(s) for s in SCENES];C=[x['comparison'] for x in info];D=[x['diagnostics'] for x in info]
    suites=C[0]['suites'];n=C[0]['queries_per_suite']
    assert all(c['suites']==suites and c['queries_per_suite']==n for c in C)
    head=lambda first:'|'+first+'|'+'|'.join(f"{x['scene']} (x{x['scale']})" for x in info)+'|'
    mppi=lambda c,k,g=None:100*(group(c['results'][k],g)[0]/group(c['results'][k],g)[1] if g else c['results'][k]['total']/c['results'][k]['n'])
    arrow=lambda vals:' -> '.join(f'{v:.1f}%' for v in vals)
    L=['# Cabinet clearance sweep: Bellman NTFields vs metric NTFields','',
       'The cabinet task (`experiments/cabinet/README.md`) in three scenes: the original cupboard and two variants with 10% and '
       '20% more clearance, where every compartment opening is 1.1x / 1.2x as wide and each compartment 1.1x / 1.2x as tall '
       '(`make_scenes.py`; panels, floor, back face and depth unchanged; the sampling box follows the same rule from the new '
       'cupboard). Each scene was prepared and trained from scratch with both methods and scored on its own frozen, certified '
       f'suites: {len(suites)} suites x {n} queries; MPPI rows count 3 planning seeds per query (1,800 executions).','',
       '## Headline (cabinet -> +10% -> +20% clearance)','',
       f"- Release MPPI, all 600 queries: Bellman {arrow([mppi(c,'bellman/matched_mppi') for c in C])}; "
       f"metric NTFields {arrow([mppi(c,'metric/matched_mppi') for c in C])}.",
       f"- RK4, all 600 queries: Bellman {arrow([mppi(c,'bellman/native_rk4') for c in C])}; metric NTFields {arrow([mppi(c,'metric/native_rk4') for c in C])}.",
       f"- Release MPPI, the four compartment suites: Bellman {arrow([mppi(c,'bellman/matched_mppi',COMPARTMENT_SUITES) for c in C])}; "
       f"metric NTFields {arrow([mppi(c,'metric/matched_mppi',COMPARTMENT_SUITES) for c in C])}"
       +(f"; straight line {arrow([100*sum(d['straight_line'][s] for s in COMPARTMENT_SUITES)/(n*len(COMPARTMENT_SUITES)) for d in D])}." if all(D) else '.'),
       f"- Bellman minus metric NTFields, release MPPI: "+' -> '.join(f"{mppi(c,'bellman/matched_mppi')-mppi(c,'metric/matched_mppi'):+.1f}" for c in C)+' points.',
       '']
    if all((c['bellman_run'],c['metric_run'])==READING_RUNS.get(x['scene']) for x,c in zip(info,C)) and all(D):
        shares=[c['culprit_group']['cupboard_panel']/c['n'] for d in D for c in d['collisions'].values()]
        L+=[line.replace('PANEL_SHARE',f'{100*min(shares):.0f}-{100*max(shares):.0f}%') for line in READING]
    L+=['## Scenes','',head(''),
       '|---|'+'---:|'*len(info)]
    L.append('|Compartment opening width (m)|'+'|'.join(f"{x['opening_width']}" for x in info)+'|')
    L.append('|Compartment heights bottom / middle / top (m)|'+'|'.join(' / '.join(f'{v:g}' for v in x['heights'].values()) for x in info)+'|')
    L.append('|Cupboard outer depth x width x height (m)|'+'|'.join(f"{x['cupboard']['depth']:g} x {x['cupboard']['width']:g} x {x['cupboard']['height']:g}" for x in info)+'|')
    L.append('|Sampling box y (m)|'+'|'.join(f"{x['sampling']['y'][0]:g} to {x['sampling']['y'][1]:g}" for x in info)+'|')
    L.append('|Sampling box z (m)|'+'|'.join(f"{x['sampling']['z'][0]:g} to {x['sampling']['z'][1]:g}" for x in info)+'|')
    L.append('|Test-suite compartment pools bottom / middle / top|'+'|'.join(' / '.join(f'{v:,}' for v in x['generation']['pool_sizes']['compartment']) for x in info)+'|')
    cert=lambda x:(sum(v.get('certified',0) for v in x['generation']['strata'].values()),sum(v['candidates'] for v in x['generation']['strata'].values()))
    L.append('|Candidates certified (direct line or RRT-Connect)|'+'|'.join(pct(*cert(x)) for x in info)+'|')
    L.append('|Frozen suites|'+'|'.join(f"`{x['tests']}/`" for x in info)+'|')
    # Success rates.
    L+=['','## Success rate, all 600 queries','',head('Method and planner'),'|---|'+'---:|'*len(info)]
    for key,label in ROWS:L.append(f'|{label}|'+'|'.join(pct(c['results'][key]['total'],c['results'][key]['n']) for c in C)+'|')
    if all(D):L.append('|Straight joint-space line (reference)|'+'|'.join(pct(d['straight_line']['total'],n*len(suites)) for d in D)+'|')
    L+=['','Compartment suites (free_to_compartment, compartment_to_free, compartment_to_compartment, same_compartment; 400 queries) '
        'and open-space suites (free_to_free, anywhere; 200 queries):','',
        '|Method and planner|'+'|'.join(f"{x['scene']} {g}" for x in info for g in ['compartment','open'])+'|','|---|'+'---:|'*(2*len(info))]
    for key,label in ROWS:L.append(f'|{label}|'+'|'.join(rate(*group(c['results'][key],g)) for c in C for g in [COMPARTMENT_SUITES,OPEN_SUITES])+'|')
    if all(D):L.append('|Straight joint-space line|'+'|'.join(rate(sum(d['straight_line'][s] for s in g),n*len(g)) for d in D for g in [COMPARTMENT_SUITES,OPEN_SUITES])+'|')
    for title,keys in [('Release MPPI per suite (same planner for both methods)',['bellman/matched_mppi','metric/matched_mppi']),
                       ('RK4 per suite',['bellman/native_rk4','metric/native_rk4'])]:
        L+=['',f'## {title}','','|Suite|'+'|'.join(f"{'Bellman' if k.startswith('bellman') else 'Metric'} {x['scene'][len('cabinet'):].lstrip('_') or 'cabinet'}" for k in keys for x in info)+'|',
            '|---|'+'---:|'*(len(keys)*len(info))]
        for s in suites:L.append(f'|{s}|'+'|'.join(rate(*c['results'][k]['per_suite'][s]) for k in keys for c in C)+'|')
    L+=['','## ntrl-demo MPPI (metric NTFields\' own planner) and the straight line per suite','',
        '|Suite|'+'|'.join(f"ntrl MPPI {x['scene'][len('cabinet'):].lstrip('_') or 'cabinet'}" for x in info)+('|'+'|'.join(f"Line {x['scene'][len('cabinet'):].lstrip('_') or 'cabinet'}" for x in info) if all(D) else '')+'|',
        '|---|'+'---:|'*(len(info)*(2 if all(D) else 1))]
    for s in suites:L.append(f'|{s}|'+'|'.join(rate(*c['results']['metric/ntrl_mppi']['per_suite'][s]) for c in C)+('|'+'|'.join(rate(d['straight_line'][s],n) for d in D) if all(D) else '')+'|')
    L+=['','## Same planner, same queries (per scene)','','|Scene|Planner|Both solve|Bellman only|Metric only|Neither|','|---|---|---:|---:|---:|---:|']
    for x,c in zip(info,C):
        for planner,label in [('native_rk4','RK4 (600)'),('matched_mppi','Release MPPI (600 x 3 seeds)')]:
            q=c['paired'][planner];L.append(f"|{x['scene']}|{label}|{q['both']}|{q['bellman_only']}|{q['metric_only']}|{q['neither']}|")
    if all(D):
        L+=['','## Diagnostics (`diagnose.py`, read-only)','','First scorer violation on paths rejected for collision (seed-0 MPPI and RK4 paths):','',
            '|Scene|Method and planner|Collision paths|Cupboard panel|Table|Self|','|---|---|---:|---:|---:|---:|']
        for x,d in zip(info,D):
            for key,c in d['collisions'].items():L.append(f"|{x['scene']}|{key}|{c['n']}|{c['culprit_group']['cupboard_panel']}|{c['culprit_group']['table']}|{c['culprit_group']['self']}|")
        L+=['','Implied speed 1/|grad_q T(q, goal)| along the certified witness paths, mean over the points inside the cupboard, against the '
            'true clearance speed there (the four compartment suites pooled; Eikonal: equal for an exact travel time):','',
            '|Scene|True speed in cupboard|Bellman implied|Metric implied|Bellman log-ratio RMSE|Metric log-ratio RMSE|','|---|---:|---:|---:|---:|---:|']
        for x,d in zip(info,D):
            rows=[d['implied_speed'][s] for s in COMPARTMENT_SUITES if d['implied_speed'][s].get('true_speed_mean_inside') is not None]
            w=[r['points_inside'] for r in rows];avg=lambda k:sum(r[k]*v for r,v in zip(rows,w))/sum(w)
            L.append(f"|{x['scene']}|{avg('true_speed_mean_inside'):.2f}|{avg('bellman_implied_mean_inside'):.2f}|{avg('metric_implied_mean_inside'):.2f}|"
                     f"{avg('bellman_log_ratio_rmse_inside'):.2f}|{avg('metric_log_ratio_rmse_inside'):.2f}|")
        L+=['','How hard the frozen compartment queries are (`diagnose.py` section 5; the four compartment suites pooled):','',
            '|Measure|'+'|'.join(x['scene'] for x in info)+'|','|---|'+'---:|'*len(info)]
        def pooled(d,key,weight='compartment_endpoints'):
            rows=[d['difficulty'][s] for s in COMPARTMENT_SUITES];return sum(r[key]*r[weight] for r in rows)/sum(r[weight] for r in rows)
        L.append('|True clearance speed at the compartment endpoints (mean)|'+'|'.join(f"{pooled(d,'compartment_endpoint_speed_mean'):.3f}" for d in D)+'|')
        L.append('|Compartment endpoint depth past the opening (m, mean)|'+'|'.join(f"{pooled(d,'compartment_endpoint_depth_mean'):.3f}" for d in D)+'|')
        L.append('|Lowest clearance speed along the certified witness path (mean)|'+'|'.join(f"{sum(d['difficulty'][s]['witness_min_speed_mean'] for s in COMPARTMENT_SUITES)/len(COMPARTMENT_SUITES):.3f}" for d in D)+'|')
        L.append('|Witness joint-space length (rad, median per suite)|'+'|'.join(' / '.join(f"{d['difficulty'][s]['witness_cspace_length_median']:.1f}" for s in COMPARTMENT_SUITES) for d in D)+'|')
        L.append('|Queries certified by RRT-Connect, not the straight line|'+'|'.join(f"{sum(d['difficulty'][s]['rrt_witnesses'] for s in COMPARTMENT_SUITES)}/{n*len(COMPARTMENT_SUITES)}" for d in D)+'|')
        L+=['','Speeds are the label `min(clip(d_obstacle, 0.005, 0.05), clip(0.5 d_self, 0.005, 0.05)) / 0.05`: 0.1 is 5 mm clearance or less, 1 is 5 cm or more.']
        L+=['','Training states with the grasp point in the cupboard (outer box) / in a compartment interior:','',
            '|States|'+'|'.join(x['scene'] for x in info)+'|','|---|'+'---:|'*len(info)]
        for key in D[0]['coverage']:
            L.append(f'|{key}|'+'|'.join(f"{100*d['coverage'][key]['cupboard']:.2f}% / {100*sum(v for k,v in d['coverage'][key].items() if k.startswith('compartment_')):.2f}%" for d in D)+'|')
    L+=['','## Training inputs and runs','','|Scene|Bellman run|Metric NTFields run|FK pool in cupboard|Speed MAE original / augmentation|Bank anchors|Metric x0 in cupboard|Metric final loss|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for x,c in zip(info,C):
        mae=' / '.join(f"{v['mae']:.4f}" for v in x['speed']['validation']);loss=float(x['metric_training']['final_checkpoint'].rsplit('ValLoss_',1)[1][:-3])
        L.append(f"|{x['scene']}|`{c['bellman_run']}`|`{c['metric_run']}`|{100*x['sampled']['pool']['cupboard_share']:.2f}%|{mae}|{x['prepared']['bank_anchors']:,}|"
                 f"{100*x['metric_data']['x0_cupboard_share']:.2f}%|{loss:.3g}|")
    L+=['','Separate evaluation runs (scene cabinet only; its pipeline evaluation stage had failed on an inverted assertion): '+
        ', '.join(f"`{c['bellman_evaluation']}`, `{c['metric_evaluation']}`" for c in C if c['bellman_evaluation']!=c['bellman_run'])+'.' if any(c['bellman_evaluation']!=c['bellman_run'] for c in C) else '',
        '','Wall time (seconds):','','|Scene|Bellman: sample / prepare / training / evaluate|Metric: data / training / evaluate|','|---|---|---|']
    for x,c in zip(info,C):
        t=c['timing'];L.append(f"|{x['scene']}|"+' / '.join(f'{v:.0f}' for v in t['bellman'].values())+f" (total {sum(t['bellman'].values()):.0f})|"
                               +' / '.join(f'{v:.0f}' for v in t['metric'].values())+f" (total {sum(t['metric'].values()):.0f})|")
    se=lambda p,m:100*(p*(1-p)/m)**.5
    L+=['','## Caveats','',
        '- Every scene has its own certified suites (endpoints valid in that scene, regions from its own compartments), so the scenes share '
        'no queries; changes across scenes mix the easier geometry with different queries. The straight-line row is the per-scene reference.',
        f'- Resolution: drawing 600 queries alone gives a success rate near 35% a standard error of about {se(.35,600):.1f} points '
        f'(about {se(.15,400):.1f} points for the 400 compartment queries near 15%); MPPI repeats are correlated per query and do not shrink it. '
        'Training-seed variation comes on top and is unmeasured.',
        '- One training seed per method and scene; planning seeds 0/1/2 are not training seeds. No significance claim.',
        '- Per-scene details: '+', '.join(f"`{x['evidence']}/COMPARISON.md`" for x in info)+'.','']
    out=ROOT/'evidence/cabinet';(out/'CLEARANCE.md').write_text('\n'.join(L))
    summary=dict(scenes=[{k:v for k,v in x.items() if k in ('scene','scale','evidence','tests','opening_width','heights','cupboard','sampling')} for x in info],
        results={x['scene']:{key:dict(success=c['results'][key]['total'],n=c['results'][key]['n'],
            compartment=group(c['results'][key],COMPARTMENT_SUITES),open=group(c['results'][key],OPEN_SUITES)) for key,_ in ROWS} for x,c in zip(info,C)},
        straight_line={x['scene']:d['straight_line'] for x,d in zip(info,D)} if all(D) else None,
        paired={x['scene']:c['paired'] for x,c in zip(info,C)},runs={x['scene']:dict(bellman=c['bellman_run'],metric=c['metric_run'],
            bellman_evaluation=c['bellman_evaluation'],metric_evaluation=c['metric_evaluation'],tests=x['frozen']['source_run']) for x,c in zip(info,C)})
    (out/'clearance.json').write_text(json.dumps(summary,indent=2)+'\n');print('\n'.join(L))


if __name__=='__main__':main()
