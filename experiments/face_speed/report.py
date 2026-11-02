"""Compare the per-piece (box-face) speed runs with their same-environment baselines on the frozen cabinet suites.

    python3 experiments/face_speed/report.py --runs registered_<id> ... [--accuracy registered_<id>] [--scene cabinet]

Every --runs entry is a face-speed pipeline run (status.json: method, cover) or a cabinet pipeline run with its
evaluation; the baseline of a method is its `release` cover run in the same environment. The pinned-environment
results of the original machine (evidence/cabinet) are listed for reference only. Writes
evidence/face_speed/COMPARISON.md and comparison.json. Standard library only.
"""
import argparse, datetime as dt, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANNERS = [('native_rk4', 'RK4 (native)'), ('matched_mppi', 'release MPPI'), ('ntrl_mppi', 'ntrl-demo MPPI')]
REFERENCE = {('bellman', 'pinned'): 'registered_20261019T041026Z-39d9dafe', ('metric', 'pinned'): 'registered_20261019T041045Z-9baa751f'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pct(s, n): return f'{s}/{n} ({100*s/n:.1f}%)' if n else '–'


def load(run):
    run = ROOT/'results'/Path(run).name; status = json.loads((run/'status.json').read_text())
    ev = json.loads((run/'evaluation/evaluation.json').read_text()); assert ev['complete'] and not ev['smoke'], run.name
    method = 'bellman' if status.get('method') == 'bellman' else 'metric'; cover = status.get('cover', 'release' if 'cover' not in status else status['cover'])
    if status.get('speed_model') == 'separate': cover += ' (separate models)'
    env = json.loads((run/'environment_override.json').read_text()) if (run/'environment_override.json').exists() else {}
    info = dict(run=run.name, method=method, cover=cover, status=status, evaluation=ev, environment=env, path=run)
    if method == 'metric' and (run/'data/complete.json').exists(): info['data'] = json.loads((run/'data/complete.json').read_text())
    elif method == 'metric' and status.get('data'): info['data'] = json.loads((Path(status['data'].replace('/clean', str(ROOT)))/'complete.json').read_text())
    if method == 'bellman':
        for name in ['speed_audit', 'bank_speed_audit', 'piece_label_audit', 'complete']:
            p = run/'prepared'/(name+'.json')
            if p.exists(): info[name] = json.loads(p.read_text())
        if (run/'training/config.json').exists(): info['training_config'] = json.loads((run/'training/config.json').read_text())
    return info


def totals(ev):
    return {m: ev['totals'][m] for m in ev['totals']}


def per_suite(ev, planner):
    out = {}
    for r in ev['scores']:
        if r['method'] == planner: out.setdefault(r['task'], [0, 0]); out[r['task']][0] += r['success']; out[r['task']][1] += r['n']
    return out


def solved(ev, planner):
    ids = {}
    for r in ev['scores']:
        if r['method'] == planner: ids[(r['task'], r['planning_seed'])] = set(r.get('success_ids', []))
    return ids


def failures(ev, planner):
    out = {}
    for r in ev['scores']:
        if r['method'] == planner:
            for k, v in r.get('failures', {}).items(): out[k] = out.get(k, 0)+v
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--accuracy'); p.add_argument('--guards', nargs='*', default=[], help='face-speed-guard runs'); p.add_argument('--scene', default='cabinet'); a = p.parse_args()
    runs = [load(r) for r in a.runs]; suffix = a.scene[len('cabinet'):]; tests = ROOT/'experiments/cabinet'/('tests'+suffix)
    manifest = json.loads((tests/'suite_manifest.json').read_text()); suites = [s['task'] for s in manifest['suites']]
    for r in runs: assert r['evaluation']['tests_manifest_sha256'] == sha(tests/'suite_manifest.json'), f"{r['run']} scored other suites"
    reference = {}
    for (method, _), name in REFERENCE.items():
        path = ROOT/'results'/name/'evaluation/evaluation.json'
        if path.exists():
            ev = json.loads(path.read_text())
            if ev.get('tests_manifest_sha256') == sha(tests/'suite_manifest.json'): reference[method] = dict(run=name, evaluation=ev)
    out = ROOT/'evidence/face_speed'; out.mkdir(parents=True, exist_ok=True)
    env = next((r['environment'] for r in runs if r['environment']), {})
    lines = [f'# Cabinet task: per-piece (box-face) speed vs the release speed', '',
             f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Frozen suites `{tests.relative_to(ROOT)}` (6 x 100 certified queries; MPPI rows count 3 planning seeds). "
             f"All runs below were made on this workstation through the host backend (bwrap, not the pinned image): {json.dumps(env.get('versions', {}))}. "
             'Rows marked *pinned* are the original machine\'s results in the pinned image (evidence/cabinet), for reference only.', '']
    # 1. Overall success.
    lines += ['## Success rate, all 600 queries', '', '|Run|Method|Speed cover|RK4 (native)|Release MPPI|Release MPPI per seed|ntrl-demo MPPI|', '|---|---|---|---:|---:|---|---:|']
    rows = []
    for r in runs:
        t = totals(r['evaluation']); rows.append(dict(run=r['run'], method=r['method'], cover=r['cover'], totals=t))
        lines.append(f"|`{r['run']}`|{r['method']}|{r['cover']}|{pct(t['native_rk4']['success'], t['native_rk4']['n'])}|{pct(t['matched_mppi']['success'], t['matched_mppi']['n'])}|{t['matched_mppi']['per_seed']}|"
                     f"{pct(t['ntrl_mppi']['success'], t['ntrl_mppi']['n']) if 'ntrl_mppi' in t else '–'}|")
    for method, ref in reference.items():
        t = totals(ref['evaluation'])
        lines.append(f"|`{ref['run']}` *pinned*|{method}|release|{pct(t['native_rk4']['success'], t['native_rk4']['n'])}|{pct(t['matched_mppi']['success'], t['matched_mppi']['n'])}|{t['matched_mppi']['per_seed']}|"
                     f"{pct(t['ntrl_mppi']['success'], t['ntrl_mppi']['n']) if 'ntrl_mppi' in t else '–'}|")
    # 2. Per suite.
    for planner, label in PLANNERS:
        cols = [r for r in runs if planner in r['evaluation']['totals']]
        if not cols: continue
        lines += ['', f'## Per suite, {label}', '', '|Suite|'+'|'.join(f"{r['method']} {r['cover']}" for r in cols)+'|', '|---|'+'---:|'*len(cols)]
        tables = [per_suite(r['evaluation'], planner) for r in cols]
        for s in suites: lines.append(f'|{s}|'+'|'.join(pct(*t.get(s, [0, 0])) for t in tables)+'|')
    # 3. Paired comparison with the same-environment baseline of the method.
    pairs = []
    for r in runs:
        if r['cover'] == 'release': continue
        base = next((b for b in runs if b['method'] == r['method'] and b['cover'] == 'release'), None)
        if base is None: continue
        for planner, label in PLANNERS:
            if planner not in r['evaluation']['totals'] or planner not in base['evaluation']['totals']: continue
            A, B = solved(r['evaluation'], planner), solved(base['evaluation'], planner); both = only_a = only_b = neither = 0
            for key in A:
                n = next(x['n'] for x in r['evaluation']['scores'] if x['method'] == planner and (x['task'], x['planning_seed']) == key)
                sa, sb = A[key], B.get(key, set()); both += len(sa & sb); only_a += len(sa-sb); only_b += len(sb-sa); neither += n-len(sa | sb)
            pairs.append(dict(variant=r['run'], cover=r['cover'], baseline=base['run'], method=r['method'], planner=planner, both=both, variant_only=only_a, baseline_only=only_b, neither=neither))
    if pairs:
        lines += ['', '## Same planner, same queries: variant vs its same-environment release baseline', '', '|Method|Cover|Planner|Both solve|Variant only|Baseline only|Neither|', '|---|---|---|---:|---:|---:|---:|']
        for q in pairs: lines.append(f"|{q['method']}|{q['cover']}|{dict(PLANNERS)[q['planner']]}|{q['both']}|{q['variant_only']}|{q['baseline_only']}|{q['neither']}|")
    # 4. Failure reasons.
    lines += ['', '## Failure reasons (scorer)', '', '|Run|Planner|'+'|'.join(['collision', 'endpoint', 'joint_limits'])+'|', '|---|---|---:|---:|---:|']
    for r in runs:
        for planner, label in PLANNERS:
            if planner in r['evaluation']['totals']:
                f = failures(r['evaluation'], planner); lines.append(f"|{r['method']} {r['cover']}|{label}|{f.get('collision', 0)}|{f.get('endpoint', 0)}|{f.get('joint_limits', 0)}|")
    # 5. Training inputs and speed networks.
    lines += ['', '## Training inputs', '']
    for r in runs:
        if r['method'] == 'metric' and 'data' in r:
            d = r['data']; cv = d.get('cover_vs_release', {})
            lines.append(f"- {r['method']} {r['cover']} (`{r['run']}`): {d['pairs']} pairs from {d['uniform_draws']} uniform draws; x0 in the cupboard {100*d['x0_cupboard_share']:.2f}%; x1 at the speed floor {100*d['x1_speed_floor_share']:.1f}%"
                         + (f"; mean speed {cv['mean_speed']:.3f} vs release {cv['mean_release_speed']:.3f}, {100*cv['share_below_release']:.1f}% of endpoints below the release speed" if cv else '')
                         + f"; final epoch loss {json.loads((r['path']/'training/progress.json').read_text())['final_loss']:.4g}; data {d['elapsed_seconds']:.0f} s.")
        elif r['method'] == 'bellman' and 'speed_audit' in r:
            v = r['speed_audit']['validation']; c = r['speed_audit']['training']; b = r.get('bank_speed_audit', {})
            held = '; '.join(f"{x['source']}: outputs MAE {x['mae']:.4f}" + (f", min speed MAE {x['min_speed']['mae']:.4f} (vs release {x['min_speed_vs_release']['mae']:.4f})" if 'min_speed' in x else '') for x in v)
            lines.append(f"- {r['method']} {r['cover']} (`{r['run']}`): {c['n_pieces']} outputs, {c['steps']} steps; held-out {held}; bank anchors {r.get('complete', {}).get('bank_anchors')}"
                         + (f"; bank-source speed MAE vs release {b['vs_release']['mae']:.4f}, vs cover truth {b['vs_cover_truth']['mae']:.4f}" if b else '') + '.')
    # 6. Wall time.
    lines += ['', '## Wall time (seconds)', '']
    for r in runs:
        t = r['status'].get('timing', {}); lines.append(f"- {r['method']} {r['cover']} [{r['environment'].get('gpus')}; {', '.join(r['environment'].get('versions', {}).get('gpus', []))}]: " + ', '.join(f'{k} {v:.0f}' for k, v in t.items()) + f"; total {r['status'].get('elapsed_seconds', 0):.0f}")
    # 7. Speed accuracy.
    accuracy = None
    if a.accuracy:
        acc = ROOT/'results'/Path(a.accuracy).name/'accuracy'; accuracy = json.loads((acc/'accuracy.json').read_text())
        lines += ['', f"## Speed network accuracy (`{Path(a.accuracy).name}`)", '', 'MAE of the predicted body speed against the release clearance speed (the scorer\'s rule); for per-piece networks also against their own cover label. Full tables: the run\'s accuracy/REPORT.md.', '',
                  '|Network|Cover|'+'|'.join(accuracy['sets'])+'|', '|---|---|'+'---:|'*len(accuracy['sets'])]
        for name, net in accuracy['networks'].items():
            lines.append(f"|`{name}`|{net['cover']}|"+'|'.join(f"{net['sets'][s]['vs_release']['mae']:.4f}"+(f" / {net['sets'][s]['vs_cover']['mae']:.4f}" if 'vs_cover' in net['sets'][s] else '') for s in accuracy['sets'])+'|')
        lines.append(''); lines.append('Cells: MAE vs release / MAE vs own cover. Sets: ' + '; '.join(f"{k} n={v['n']}, release speed at floor {100*v['share_release_at_floor']:.0f}%" for k, v in accuracy['sets'].items()) + '.')
    # 8. Step-filter evaluations.
    guards = []
    for g in a.guards:
        path = ROOT/'results'/Path(g).name/'evaluation/evaluation.json'; ev = json.loads(path.read_text()); assert ev['complete'] and ev['tests_manifest_sha256'] == sha(tests/'suite_manifest.json'), g
        act = list(ev.get('guard_reports', {}).values()); guards.append(dict(run=Path(g).name, field=ev['run'], method=ev['method'], field_cover=ev.get('field_cover') or 'release', guard=ev['guard'], totals=ev['totals'],
            activity=dict(active=sum(x['filter_active_frac'] for x in act)/max(1, len(act)), change=sum(x['mean_relative_change'] for x in act)/max(1, len(act)), retries=sum(x['retries'] for x in act), stalled=sum(x['stalled_steps'] for x in act)) if act else None))
    if guards:
        lines += ['', '## Step filter at planning time (cspace_face_filter Section 3)', '', 'The trained field is unchanged; each planner step is projected onto the half-spaces of the active pieces (`guard.py`). RK4: the native RK4 rollout; MPPI: the ntrl-demo MPPI settings for either field (the release MPPI planner is frozen and not filtered). "converged" counts rollouts that reached the goal; the scorer decides success.', '',
                  '|Field (run)|Filter cover|Rows|alpha|floor|RK4 success|RK4 converged|MPPI success (3 seeds)|MPPI converged|Filter active|Mean step change|Halvings|Stalled steps|', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g in sorted(guards, key=lambda x: (x['method'], x['field_cover'], x['guard']['rows'], x['guard']['cover'], x['guard']['floor'])):
            t = g['totals']; r, m = t['guarded_rk4'], t.get('guarded_ntrl_mppi'); act = g['activity']
            lines.append(f"|{g['method']} {g['field_cover']} (`{g['field']}`)|{g['guard']['cover'] if g['guard']['rows'] != 'none' else '–'}|{g['guard']['rows']}|{g['guard']['alpha']}|{g['guard']['floor']}|{pct(r['success'], r['n'])}|{r['converged']}|"
                         + (f"{pct(m['success'], m['n'])}|{m['converged']}|" if m else '–|–|') + (f"{100*act['active']:.0f}%|{act['change']:.3f}|{act['retries']}|{act['stalled']}|" if act else '–|–|–|–|'))
        lines += ['', '|Field|Filter|RK4 per suite|MPPI per suite|', '|---|---|---|---|']
        for g in sorted(guards, key=lambda x: (x['method'], x['field_cover'], x['guard']['rows'], x['guard']['cover'], x['guard']['floor'])):
            ev = json.loads((ROOT/'results'/g['run']/'evaluation/evaluation.json').read_text()); rk, mp = per_suite(ev, 'guarded_rk4'), per_suite(ev, 'guarded_ntrl_mppi')
            lines.append(f"|{g['method']} {g['field_cover']}|{g['guard']['rows']} {g['guard']['cover'] if g['guard']['rows'] != 'none' else ''} a{g['guard']['alpha']} f{g['guard']['floor']}|"+', '.join(f"{k} {v[0]}/{v[1]}" for k, v in rk.items())+'|'+', '.join(f"{k} {v[0]}/{v[1]}" for k, v in mp.items())+'|')
    lines += ['', 'One training seed per run; planning seeds 0/1/2 are not training seeds; no significance claim. Modified environment (host backend), see environment_override.json in each run.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (out/'comparison.json').write_text(json.dumps(dict(generated=dt.datetime.now(dt.timezone.utc).isoformat(), scene=a.scene, runs=rows, pairs=pairs, reference={k: v['run'] for k, v in reference.items()},
                                                       accuracy=a.accuracy, guards=guards, environment=env), indent=2)+'\n')
    print((out/'COMPARISON.md').read_text())


if __name__ == '__main__': main()
