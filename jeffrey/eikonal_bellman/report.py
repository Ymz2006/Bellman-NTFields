"""Tables of the eikonal-foundation Bellman-extension experiment from registered runs (markdown on stdout).

    python3 jeffrey/eikonal_bellman/report.py --state cab_state.json [nw_state.json ...] [--fit-ranges 0,0.24 0.24,0.72 0.72,1.6]
The state files are the queue runner's label -> run id maps. For every training run: planning totals, per-level target
and model diagnostics against the ground truth (chain.py --truth), fit-stage summaries (segment_dp/fit.py) when a
fit-<label> stage exists.
"""
import argparse, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2]


def load(run):
    base = HERE/'results'/f'registered_{run}'
    out = dict(status=json.loads((base/'status.json').read_text()) if (base/'status.json').exists() else None)
    for name in ('progress', 'config'):
        p = base/'training'/f'{name}.json'
        out[name] = json.loads(p.read_text()) if p.exists() else None
    return out


def planning_table(states):
    lines = ['|Label|Scene|Scheme|Models|Split|Weights|RK4 /600|MPPI /1800 (seeds)|Train min|Eval min|', '|---|---|---|---|---|---|---:|---|---:|---:|']
    for label, row in states.items():
        if label.startswith(('fit-', 'long')) or not row.get('id'): continue
        r = load(row['id']); c = r['config']; s = r['status']
        if not c: continue
        tot = (s or {}).get('totals'); scene = 'cabinet' if (s or {}).get('settings', {}).get('scene') else 'nowall'
        rk = tot['native_rk4']['success'] if tot else '—'; mp = f"{tot['matched_mppi']['success']} ({'/'.join(map(str, tot['matched_mppi']['per_seed']))})" if tot else '—'
        tm = (s or {}).get('timing', {})
        if 'weights' not in c:      # a foundation (eikonal) run
            mode = c.get('mode', 'eikonal'); extra = f" σ={c['sigma']:g}, {c.get('augment_rows', 0)} aug rows" if 'sigma' in c else ''
            lines.append(f"|{label}|{scene}|foundation ({mode}{extra})|1|—|eikonal|{rk}|{mp}|{tm.get('training', 0)/60:.0f}|{tm.get('evaluate', 0)/60:.0f}|"); continue
        w = c['weights']; weights = f"{w['bellman']:g}/{w['eikonal']:g}/{w['direction']:g}"+(f" +reh{c['rehearsal']}" if c.get('rehearsal') else '')
        split = c['split']+(' +clear' if c.get('clearance') else '')+(f" rho={c['support_rho']:.3f}" if c.get('support_rho') else '')
        lines.append(f"|{label}|{scene}|{c['scheme']}|{c['models']}{' x2' if c['estimator'] == 'double' else ''}|{split}|{weights}|{rk}|{mp}|{tm.get('training', 0)/60:.0f}|{tm.get('evaluate', 0)/60:.0f}|")
    return '\n'.join(lines)


def suite_table(states, scene):
    """Release MPPI (3 planning seeds) per suite and RK4 per suite of every evaluated run of one scene."""
    tasks = dict(cabinet=['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere'],
                 nowall=['mixed', 'screwdriver', 'cutting', 'hammering', 'stirring', 'wiping'])[scene]
    lines = ['|Label|'+'|'.join(t.replace('_to_', '→').replace('compartment', 'c').replace('free', 'f') for t in tasks)+'|MPPI|RK4 per suite|RK4|', '|---|'+'---:|'*(len(tasks)+1)+'---|---:|']
    for label, row in states.items():
        if label.startswith(('fit-', 'long')) or not row.get('id'): continue
        base = HERE/'results'/f"registered_{row['id']}"/'evaluation/evaluation.json'
        if not base.exists(): continue
        s = json.loads((HERE/'results'/f"registered_{row['id']}"/'status.json').read_text())
        if ('cabinet' if s.get('settings', {}).get('scene') else 'nowall') != scene: continue
        ev = json.loads(base.read_text())
        mp = [sum(r['success'] for r in ev['scores'] if r['method'] == 'matched_mppi' and r['task'] == t) for t in tasks]
        rk = [sum(r['success'] for r in ev['scores'] if r['method'] == 'native_rk4' and r['task'] == t) for t in tasks]
        lines.append(f"|{label}|"+'|'.join(map(str, mp))+f"|{sum(mp)}|{', '.join(map(str, rk))}|{sum(rk)}|")
    return '\n'.join(lines)


def level_table(states, metric):
    """Per-level diagnostic: metric in target_over_truth, model_over_truth, target_abs_log_error, model_abs_log_error, detour, held_out, infeasible."""
    runs = [(l, load(r['id'])) for l, r in states.items() if r.get('id') and not l.startswith(('fit-', 'long'))]
    runs = [(l, r) for l, r in runs if r['progress'] and r['progress'].get('levels')]
    if not runs: return ''
    levels = max(len(r['progress']['levels']) for _, r in runs); edges = runs[0][1]['config']['edges']
    head = '|Level (gap)|'+'|'.join(l for l, _ in runs)+'|'; lines = [head, '|---|'+'---:|'*len(runs)]
    for k in range(levels):
        cells = []
        for _, r in runs:
            lv = r['progress']['levels']; cell = '—'
            if k < len(lv):
                x = lv[k]; t = x.get('truth') or {}
                if metric == 'target_over_truth': v = t.get('target_over_truth_median')
                elif metric == 'target_abs_log_error': v = t.get('target_abs_log_error')
                elif metric == 'model_over_truth': v = x['fits'][0].get('model_over_truth_median')
                elif metric == 'model_abs_log_error': v = x['fits'][0].get('model_abs_log_error')
                elif metric == 'detour': v = x.get('detour_median')
                elif metric == 'held_out': v = x['fits'][0]['held_out']['mean_abs_log_ratio']
                elif metric == 'infeasible': v = x.get('infeasible')
                elif metric == 'target_over_gap': v = x.get('target_over_gap_median')
                elif metric == 'single_vs_double': v = t.get('single_estimator_target_over_truth_median')
                else: v = None
                cell = '—' if v is None else f'{v:.3f}'
            cells.append(cell)
        seg = runs[0][1]['config']['levels'][k] if k < len(runs[0][1]['config']['levels']) else ['?', '?']
        lines.append(f'|{k+1} ({seg[0]:.2f}–{seg[1]:.2f})|'+'|'.join(cells)+'|')
    return '\n'.join(lines)


def fit_summary(states, ranges):
    lines = []
    for label, row in states.items():
        if not label.startswith('fit-') or not row.get('id'): continue
        p = HERE/'results'/f"registered_{row['id']}"/'fit.json'
        if not p.exists(): continue
        fit = json.loads(p.read_text()); edges = np.array(fit['edges'])
        for f, rowf in fit['fields'].items():
            if f in ('straight', 'euclid'): continue
            cells = []
            for metric in ('rel_error', 'bias', 'spearman', 'cosine'):
                for a, b in ranges:
                    sel = [s for s in range(len(edges)-1) if edges[s] >= a-1e-9 and edges[s+1] <= b+1e-9 and rowf[metric][s] is not None]
                    cells.append(f"{np.mean([rowf[metric][s] for s in sel]):.3f}" if sel else '—')
            lines.append(f'|{f}|'+'|'.join(cells)+'|')
    if not lines: return ''
    head = '|Field|'+'|'.join(f'{m} {a:g}–{b:g}' for m in ('rel. error', 'bias', 'Spearman', 'cosine') for a, b in ranges)+'|'
    return '\n'.join([head, '|---|'+'---:|'*(4*len(ranges))]+lines)


def run_index(states_by_scene):
    lines = ['|Scene/label|Run|Kind|', '|---|---|---|']
    for scene, states in states_by_scene.items():
        for label, row in states.items():
            if not row.get('id'): continue
            if label.startswith('fit-'): kind = 'fit stage'
            elif label.startswith('long'): kind = 'long-query stage'
            else:
                st = (HERE/'results'/f"registered_{row['id']}"/'status.json')
                has_totals = st.exists() and json.loads(st.read_text()).get('totals') is not None
                kind = 'training + evaluation' if has_totals else ('training only (diagnostic, no evaluation)' if st.exists() else 'pre-existing registered run')
            if row.get('state') != 'succeeded': kind += f" ({row.get('state')})"
            lines.append(f"|{scene}/{label}|`{row['id']}`|{kind}|")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--state', type=Path, nargs='+', required=True); p.add_argument('--fit-ranges', nargs='*', default=['0,0.24', '0.24,0.72', '0.72,1.6']); a = p.parse_args()
    ranges = [tuple(float(x) for x in r.split(',')) for r in a.fit_ranges]
    for path in a.state:
        states = json.loads(path.read_text()); print(f'\n# {path.stem}\n\n## Planning\n'); print(planning_table(states))
        scene = 'cabinet' if 'cab' in path.stem else 'nowall'; print(f'\n## Per suite ({scene})\n'); print(suite_table(states, scene))
        for metric in ('target_over_truth', 'model_over_truth', 'target_abs_log_error', 'model_abs_log_error', 'detour', 'held_out', 'infeasible', 'single_vs_double'):
            t = level_table(states, metric)
            if t: print(f'\n## Per level: {metric}\n'); print(t)
        f = fit_summary(states, ranges)
        if f: print('\n## Fit to ground truth (fit stages)\n'); print(f)
    print('\n## Run index\n'); print(run_index({('cabinet' if 'cab' in path.stem else 'nowall'): json.loads(path.read_text()) for path in a.state}))


if __name__ == '__main__': main()
