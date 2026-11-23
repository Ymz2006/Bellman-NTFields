"""Report page of the LBS arm study: report/lbs_arm.html from report/runs.json (report.py).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs_arm/page.py

Chart code and style sheet are jeffrey/lift_scale/page_assets. Published as the artifact "2026-11-20 LBS Cabinet Arm".
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
LS = ROOT / 'jeffrey/lift_scale'
_spec = importlib.util.spec_from_file_location('lift_scale_pagelib', LS / 'pagelib.py')
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
f, hbar, line, table, details, chart, esc = L.f, L.hbar, L.line, L.table, L.details, L.chart, L.esc

VIEWER_URL = 'https://claude.ai/artifact/JzeY8pHydwecHDoowyJXnz'
TITLE_TAG = '2026-11-20 LBS Cabinet Arm'
TASK = dict(free_to_free='free → free', free_to_compartment='free → compartment', compartment_to_free='compartment → free',
            compartment_to_compartment='compartment → another compartment', same_compartment='inside one compartment', anywhere='anywhere → anywhere')
EXTRA_CSS = '''
.status { display: grid; gap: 4px; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 16px; margin: 10px 0 4px; font-size: 14.5px; }
.status b { font-weight: 600; }
.sumtbl { overflow-x: auto; margin: 10px 0 16px; }
.sumtbl table { min-width: 0; width: 100%; max-width: 860px; font-size: 14.5px; }
.sumtbl td, .sumtbl th { padding: 6px 10px; }
'''
ORDER = ['lbs', 'lbscup', 'lbsd', 'lbsdc', 'lbsg04', 'lbsg2', 'lbsg6', 'lbsf025', 'lbsf05', 'lbsf2', 'lbsf4', 'lbsfs', 'noL', 'noB', 'noS', 'liftonly', 'ntrl', 'ntrlc']


def run_name(r, long=False):
    a = r['arm']
    if not a['L'] and not a['B'] and not a['S']:
        return 'ntrl-demo, this study (control)' + (', half of the pairs from the cupboard' if r.get('data_cup', 0) > 0 else '')
    parts = [k for k in 'LBS' if a[k]]
    name = 'LBS' if len(parts) == 3 else {'BS': 'no lift', 'LS': 'no Bellman', 'LB': 'no segment', 'L': 'lift only'}.get(''.join(parts), '+'.join(parts))
    bits = []
    if a['L']:
        bits.append(f'h range {r["lift"]["h_range"]:.2f}')
    fr = r['freq']
    if any(abs(v - 1) > 1e-9 for v in fr):
        bits.append(f'joint frequency × {fr[0]:g}' if max(fr) - min(fr) < 1e-9 else 'joint frequency by sensitivity')
    if r.get('data_cup', 0) > 0:
        bits.append('half of the pairs from the cupboard')
    if r.get('pair_cup', 0) > 0:
        bits.append('segment endpoints in the cupboard')
    return name + (', ' + ', '.join(bits) if bits else '')


def slot(r):
    a = r['arm']
    if not (a['L'] or a['B'] or a['S']):
        return 3
    return 4 if r['label'] == 'lbs' else 5


def pct(a, b):
    return 100.0 * a / b


def main():
    D = json.loads((OUT / 'runs.json').read_text())
    SEED = 87358
    scored = sorted([r for r in D['runs'] if 'official' in r], key=lambda r: (ORDER.index(r['label']) if r['label'] in ORDER else 99, r['seed']))
    every = [r for r in scored if r['seed'] == SEED]
    reps_other = [r for r in scored if r['seed'] != SEED and abs(r['freq'][0] - 1) > 1e-9]   # other seeds at another frequency: named where they are used
    reps = [r for r in scored if r['seed'] != SEED and abs(r['freq'][0] - 1) < 1e-9]                     # the same arms with other training seeds
    twelve = sorted([r for r in D['runs'] if r.get('scene') == 'real_world_12dof' and r['seed'] == SEED and r['label'].startswith('rw12b')], key=lambda r: any(r['arm'].values()))
    short12 = [r for r in D['runs'] if r.get('scene') == 'real_world_12dof' and r['label'].startswith('rw12_')]      # trained on the first pair set, whose pairs are too short for the queries
    runs = [r for r in every if r.get('scene', 'cabinet') == 'cabinet']
    real = [r for r in every if r.get('scene', 'cabinet') != 'cabinet' and r['label'] in ('rw_ntrl', 'rw_lbs', 'rw_lift')]
    leafx = [r for r in every if r.get('scene', 'cabinet') != 'cabinet' and r['label'].startswith('rw_lbs_')]       # the segment-loss versions (leaf experiment)
    base = {b['key']: b for b in D['baselines']}
    n_line = sum(D['line_feasible'].values())
    C = {}
    rows_all = [('straight line (control)', pct(n_line, 600), 6, 'no planner: the 600 queries joined by a straight line in joint space'),
                ('ntrl-demo, best earlier setting', pct(base['ntrl_best']['official']['ntrl_mppi']['success'], 1800), 1, 'registered run 664b5d50, its own planner settings'),
                ('P-NTFields', pct(base['pnt']['official']['ntrl_mppi']['success'], 1800), 2, 'registered run 53c1dd59')]
    rows_det = [('straight line (control)', 0.0, 6, 'by definition'),
                ('ntrl-demo, best earlier setting', pct(*base['ntrl_best']['official']['ntrl_mppi']['detour']), 1, ''),
                ('P-NTFields', pct(*base['pnt']['official']['ntrl_mppi']['detour']), 2, '')]
    for r in runs:
        o = r['official']['ntrl_mppi']
        rows_all.append((run_name(r), pct(o['success'], o['n']), slot(r), f'field {r["id"][-8:]}, evaluation {r["eval"][-8:]}'))
        rows_det.append((run_name(r), pct(*o['detour']), slot(r), ''))
    leg = [('straight line (control)', 6), ('ntrl-demo, earlier run', 1), ('P-NTFields, earlier run', 2), ('ntrl-demo, this study', 3), ('LBS', 4), ('LBS, other knob settings', 5)]
    sub = 'The cabinet evaluator of the earlier runs: ntrl-demo’s MPPI, planning seeds 0, 1 and 2, release scorer. Training seed 87358.'
    C['all'] = hbar('Cabinet, 6-joint arm: queries solved', sub, [('all 600 queries (1,800 runs)', rows_all), ('the 398 queries that need a detour (1,194 runs)', rows_det)], legend=leg, value_name='solved')
    groups = []
    for t in D['tasks']:
        rows = [('straight line (control)', float(D['line_feasible'][t]), 6, ''),
                ('ntrl-demo, best earlier setting', base['ntrl_best']['official']['ntrl_mppi']['per_task'][t] / 3, 1, ''),
                ('P-NTFields', base['pnt']['official']['ntrl_mppi']['per_task'][t] / 3, 2, '')]
        for r in runs:
            rows.append((run_name(r), r['official']['ntrl_mppi']['per_task'][t] / 3, slot(r), ''))
        groups.append((TASK[t], rows))
    C['tasks'] = hbar('Queries solved by suite', 'Each suite has 100 queries; the bars are the percentage solved over the three planning seeds. Regions are those of the grasp point.', groups, legend=leg, value_name='solved')
    ser = []
    dashes = [None, '6 3', '2 3', '8 3 2 3', '1 3', '10 4']
    k = 0
    for r in runs:
        if not r['course']:
            continue
        s = slot(r)
        dash = None
        if s == 5:
            k += 1
            dash = dashes[k % len(dashes)]
        ser.append((run_name(r), s, [(c['epoch'], pct(c['success'], c['runs'])) for c in r['course']], dash))
    if ser:
        C['course'] = line('Queries solved during training', 'Planning seed 0, the network of that epoch, all 600 queries, scored by this study’s copy of the scorer’s rule. The segment loss starts at epoch 2,000.',
                           ser, xlabel='epoch', ylabel='solved %', xticks=[0, 1000, 2000, 3000, 4000, 5000], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, xmin=0, xmax=5000, digits=1, xname='epoch', height=300,
                           refs=[(pct(n_line, 600), 'straight line')])
        C['course_detour'] = line('Detour queries solved during training', 'The 398 queries whose straight line collides; planning seed 0.',
                                  [(n, s, [(c['epoch'], pct(*c['detour'])) for c in r['course']], d) for (n, s, _, d), r in zip(ser, [r for r in runs if r['course']])],
                                  xlabel='epoch', ylabel='solved %', xticks=[0, 1000, 2000, 3000, 4000, 5000], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, xmin=0, xmax=5000, digits=1, xname='epoch', height=300)
    H = []
    A = H.append
    ch = lambda key: chart(key) if key in C else ''  # noqa: E731
    lbs_runs = [r for r in runs if r['arm']['L'] and r['arm']['B'] and r['arm']['S']]
    ctl = next((r for r in runs if not any(r['arm'].values())), None)
    best = max(lbs_runs, key=lambda r: r['official']['ntrl_mppi']['success']) if lbs_runs else None
    base_lbs_ = next((r for r in lbs_runs if r['label'] == 'lbs'), None)
    if best is not None and base_lbs_ is not None and best['official']['ntrl_mppi']['success'] - base_lbs_['official']['ntrl_mppi']['success'] < 36:   # within 2 points of the first run (one seed each): report the first run
        best = base_lbs_
    A('<header><h1>LBS on the cabinet arm</h1>'
      '<p class="lede">Lift, Bellman and segment (LBS) on the 6-joint UR5e in the cupboard scene, against the earlier metric NTFields (ntrl-demo) and P-NTFields runs on the same 600 frozen queries.</p></header>')
    A(f'<p><b>See the paths:</b> a <a href="{VIEWER_URL}">3-D viewer</a> shows the arm on every test query, following the certified path, the LBS plan and the ntrl-demo plan.</p>')
    pend = D['pending']
    st = f'<div><b>{len(scored)}</b> runs of this study are trained and scored by the cabinet evaluator' + (f', {len(twelve)} on the two-arm scene by this study’s scorer' if twelve else '') + '.</div>'
    if any(p.get('epoch') is not None for p in pend):
        st += '<div>Training now: ' + ', '.join(f'{p["label"]} (epoch {p["epoch"]:,} of 5,000)' for p in pend if p.get('epoch') is not None) + '.</div>'
    if any(p.get('epoch') is None for p in pend):
        st += '<div>Queued: ' + ', '.join(p['label'] for p in pend if p.get('epoch') is None) + '.</div>'
    A(f'<div class="status">{st}</div>')
    A('<h2 id="result">Result</h2>')
    if best and ctl:
        ob, oc = best['official']['ntrl_mppi'], ctl['official']['ntrl_mppi']
        nb, pb = base['ntrl_best']['official']['ntrl_mppi'], base['pnt']['official']['ntrl_mppi']
        A(f'<p><b>{esc(run_name(best))} solves {f(pct(ob["success"], ob["n"]))} % of the runs</b> ({ob["success"]:,} of {ob["n"]:,}); ntrl-demo trained on the same data solves {f(pct(oc["success"], oc["n"]))} %, '
          f'the earlier ntrl-demo run {f(pct(nb["success"], nb["n"]))} %, P-NTFields {f(pct(pb["success"], pb["n"]))} %, and a straight line {f(pct(n_line, 600))} %. '
          f'On the queries that need a detour: LBS {f(pct(*ob["detour"]))} %, ntrl-demo here {f(pct(*oc["detour"]))} %, the earlier ntrl-demo {f(pct(*nb["detour"]))} %, P-NTFields {f(pct(*pb["detour"]))} %.</p>')
    elif ctl:
        oc = ctl['official']['ntrl_mppi']
        A(f'<p>So far only the control is scored: ntrl-demo trained on this study’s data solves {f(pct(oc["success"], oc["n"]))} % of the runs ({oc["success"]:,} of {oc["n"]:,}), '
          f'{f(pct(*oc["detour"]))} % of the detour runs. The LBS runs are training.</p>')
    A(ch('all'))
    rows = [['straight line (control)', '–', '–', f'{n_line} / 600 ({f(pct(n_line, 600))} %)', '0', str(D['line_feasible']['compartment_to_compartment']) + ' / 100', '–']]
    for key in ('ntrl_best', 'ntrl_base', 'pnt'):
        b = base[key]
        o = b['official']['ntrl_mppi']
        rows.append([esc(b['name']), '–', '–', f'{o["success"]:,} ({f(pct(o["success"], o["n"]))} %)', f'{o["detour"][0]} ({f(pct(*o["detour"]))} %)', f'{o["per_task"]["compartment_to_compartment"]} / 300',
                     f'{b["official"]["native_rk4"]["success"]} / 600'])
    for r in runs:
        o = r['official']['ntrl_mppi']
        fr = r['freq']
        rows.append([esc(run_name(r)), f'{r["lift"]["h_range"]:.2f}' if r['arm']['L'] else '–', (f'× {fr[0]:g}' if max(fr) - min(fr) < 1e-9 else 'by sensitivity'),
                     f'{o["success"]:,} ({f(pct(o["success"], o["n"]))} %)', f'{o["detour"][0]} ({f(pct(*o["detour"]))} %)', f'{o["per_task"]["compartment_to_compartment"]} / 300',
                     f'{r["official"]["native_rk4"]["success"]} / 600'])
    A('<div class="sumtbl">' + table(['method', 'range of h (joints: 1)', 'joint Fourier frequency', 'solved of 1,800 runs', 'detour runs solved of 1,194', 'compartment → another', 'RK4 descent'], rows, left=1) + '</div>')
    A('<p class="envnote">One training seed per row. The earlier ntrl-demo and P-NTFields rows are registered runs with their own training data and planner settings; '
      'the ntrl-demo row of this study uses this study’s data, speed model and planner settings, so it is the like-for-like control.</p>')
    cab = [r for r in runs if not r.get('data_cup', 0) and not r.get('pair_cup', 0)]
    one = lambda r: max(r['freq']) - min(r['freq']) < 1e-9  # noqa: E731
    val = lambda r: (pct(r['official']['ntrl_mppi']['success'], r['official']['ntrl_mppi']['n']), slot(r), f'detour runs {f(pct(*r["official"]["ntrl_mppi"]["detour"]))} %')  # noqa: E731
    g_gain = sorted([r for r in cab if r['arm']['B'] and r['arm']['S'] and one(r) and abs(r['freq'][0] - 1) < 1e-9], key=lambda r: r['lift']['h_range'] if r['arm']['L'] else 0)
    g_freq = sorted([r for r in lbs_runs if r in cab and abs(r['lift']['gain'] - 1) < 1e-9], key=lambda r: r['freq'][0] if one(r) else 99)
    g_data = [r for r in runs if (r.get('data_cup', 0) or r.get('pair_cup', 0)) or r['label'] in ('lbs', 'ntrl')]
    dname = lambda r: ('LBS' if r['arm']['L'] else 'ntrl-demo') + (', half the pairs in the cupboard' if r.get('data_cup', 0) else '') + (', endpoints in the cupboard' if r.get('pair_cup', 0) else '') \
        + ('' if r.get('data_cup', 0) or r.get('pair_cup', 0) else ', first training set')  # noqa: E731
    plain = [r for r in cab if one(r) and abs(r['freq'][0] - 1) < 1e-9 and (not r['arm']['L'] or abs(r['lift']['gain'] - 1) < 1e-9)]
    akey = lambda r: ''.join(k for k in 'LBS' if r['arm'][k])  # noqa: E731
    aname = lambda r: {'LBS': 'LBS (lift, Bellman, segment)', 'BS': 'no lift', 'LS': 'no Bellman', 'LB': 'no segment', 'L': 'lift only', '': 'none of the three: ntrl-demo (control)'}.get(akey(r), akey(r))  # noqa: E731
    g_abl = [r for k in ('LBS', 'BS', 'LS', 'LB', 'L', '') for r in plain if akey(r) == k][:6]
    if len(g_gain) > 1:
        C['knobs'] = hbar('The knobs: range of h, joint Fourier frequency, where the samples come from', 'Queries solved of the 1,800 runs; one knob changed at a time from the first LBS run (h range 0.12, frequency × 1).',
                          [('range of h (every joint has range 1)', [((f'h range {r["lift"]["h_range"]:.2f}' if r['arm']['L'] else 'no lift (Bellman and segment only)'), *val(r)) for r in g_gain]),
                           ('joint Fourier frequency', [((f'× {r["freq"][0]:g}' if one(r) else 'by joint sensitivity'), *val(r)) for r in g_freq]),
                           ('where the training samples come from', [(dname(r), *val(r)) for r in g_data])]
                          + ([('components: one removed at a time, then the lift alone', [(aname(r), *val(r)) for r in g_abl])] if len(g_abl) > 2 else []), value_name='solved')
        A('<h2 id="knobs">The knobs</h2>')
        gl = [r for r in g_gain if r['arm']['L']]
        txt = ('<p><b>Range of h.</b> ' + ', '.join(f'{r["lift"]["h_range"]:.2f}: {f(val(r)[0])} %' for r in gl) + '. '
               + (f'The best is {max(gl, key=lambda r: val(r)[0])["lift"]["h_range"]:.2f}, the base setting; a narrower and a wider h each lose a few points. ' if gl and max(gl, key=lambda r: val(r)[0]) is base_lbs_ else '')
               + 'The values 0.29 and 0.71 are the base range times √6 and times 6 (h scaled up with the six degrees of freedom); 0.05 is the base range divided by √6.')
        nl = next((r for r in g_gain if not r['arm']['L']), None)
        if nl is not None:
            txt += f' <b>Without the lift</b> (h removed, Bellman and segment kept) the run solves {f(val(nl)[0])} %' + (f', below ntrl-demo on the same data ({f(val(ctl)[0])} %).' if ctl is not None and val(nl)[0] < val(ctl)[0] else '.')
        A(txt + '</p>')
        ab = {akey(r): r for r in g_abl}
        if 'LBS' in ab and ('LB' in ab or 'LS' in ab):
            txt = f'<p><b>Components.</b> One removed at a time from the first LBS run ({f(val(ab["LBS"])[0])} %): ' + ', '.join(f'{aname(ab[k])} {f(val(ab[k])[0])} %' for k in ('BS', 'LS', 'LB') if k in ab) + '.'
            if 'LB' in ab and abs(val(ab['LB'])[0] - val(ab['LBS'])[0]) < 2:
                txt += ' The segment loss adds nothing here: the run without it is within one point, which fits the training curves (flat from epoch 500, before the segment loss starts at 2,000).'
            if 'LS' in ab:
                dlt = val(ab['LBS'])[0] - val(ab['LS'])[0]
                txt += (' Without the Bellman update the run loses ' + f(dlt) + ' points, so the gain needs the lift and the Bellman update together.' if dlt > 5 else
                        ' Without the Bellman update the result is within ' + f(abs(dlt)) + ' points, so it can be dropped too.' if abs(dlt) <= 5 else ' Without the Bellman update the run is ' + f(-dlt) + ' points higher.')
            else:
                txt += ' The run without the Bellman update is still training.'
            if 'L' in ab:
                dl = val(ab['L'])[0] - val(ab['LBS'])[0]
                txt += (f' <b>The lift alone</b> (no Bellman update, no segment loss) solves {f(val(ab["L"])[0])} %' + (f', {f(abs(dl))} points from all three together: on this scene the lift gives nearly all of the gain, and the two other terms together add a point or two at most (one training seed each).' if abs(dl) < 2 else
                        f', {f(abs(dl))} points ' + ('below' if dl < 0 else 'above') + ' all three together' + (', so the lift needs one of the two other terms.' if dl < -5 else '.')))
            else:
                txt += ' Each of the two terms can be dropped on its own; the lift alone, with neither, is queued to say whether one of them is needed at all.'
            A(txt + '</p>')
        if len(g_freq) > 1:
            txt = '<p><b>Joint Fourier frequency.</b> ' + ', '.join((f'× {r["freq"][0]:g}' if one(r) else 'by joint sensitivity') + f': {f(val(r)[0])} %' for r in g_freq) + '.'
            hi = [r for r in g_freq if one(r) and r['freq'][0] > 1 + 1e-9]
            if hi and all(val(r)[0] < val(base_lbs_)[0] - 5 for r in hi):
                txt += ' A finer frequency breaks the field instead of sharpening it.'
                ends = {}
                for r in [base_lbs_] + hi:
                    q = OUT / f'stalls_{r["id"][-8:]}.json'
                    if q.exists() and 'ends' in json.loads(q.read_text()):
                        ends[r['label']] = (r, json.loads(q.read_text())['ends'])
                worst = max((v for k, v in ends.items() if k != 'lbs'), key=lambda v: v[0]['freq'][0], default=None)
                if worst is not None and 'lbs' in ends:
                    (rw, ew), (_, e0) = worst, ends['lbs']
                    txt += (f' At × {rw["freq"][0]:g}, {ew["count"]} of 600 plans (planning seed 0) do not reach the goal, against {e0["count"]} at × 1. They stop at local minima of the field: for {ew["local"]["3"]["none_lower"]} of them '
                            f'no point within three planner steps has a lower time to the goal ({e0["local"]["3"]["none_lower"]} of {e0["count"]} at × 1). The median stop is {f(ew["remaining_over_line"][2], 2)} times the start-to-goal distance '
                            f'from the goal ({f(e0["remaining_over_line"][2], 2)} at × 1), while the field’s time to the goal has fallen to {f(100 * ew["T_end_over_T_start"][2], 0)} % of its starting value, '
                            'so the field falls along the path without the arm getting closer.')
            lo = [r for r in g_freq if one(r) and r['freq'][0] < 1 - 1e-9]
            if lo and all(abs(val(r)[0] - val(base_lbs_)[0]) < 3 for r in lo):
                txt += ' A coarser frequency (' + ', '.join(f'× {r["freq"][0]:g}' for r in lo) + ') leaves the result where it is, so the result does not rest on detail at the scale of the base frequency either.'
            A(txt + '</p>')
        dn = {r['label']: r for r in g_data}
        if 'lbsd' in dn and 'ntrlc' in dn and base_lbs_ is not None and ctl is not None:
            A(f'<p><b>Training samples.</b> A training set with {f(100 * dn["lbsd"]["data_cup"], 0)} % of its pairs starting with the hand in the cupboard (3.8 % in the first set) gives LBS {f(val(dn["lbsd"])[0])} % '
              f'(first set {f(val(base_lbs_)[0])} %) and ntrl-demo {f(val(dn["ntrlc"])[0])} % (first set {f(val(ctl)[0])} %). Drawing half of the segment loss’s endpoints inside the cupboard gives '
              + (f'{f(val(dn["lbscup"])[0])} %' if 'lbscup' in dn else 'no result yet') + '. Neither moves the result.</p>')
        A(chart('knobs'))
    A('<h2 id="suites">By suite</h2>')
    A('<p>The earlier methods solve the suites in which most queries are already solved by a straight line (free → free, anywhere → anywhere) and almost none of the others. '
      'Compartment → another compartment always needs the arm to leave one compartment and enter another.</p>')
    A(ch('tasks'))
    A('<h2 id="course">During training</h2>')
    A(f'<div class="grid2">{ch("course")}{ch("course_detour")}</div>')
    if runs:
        rows = []
        for r in runs:
            pl = r['own']['plan']
            n = sum(v['runs'] for v in pl.values())
            ok = sum(v['success'] for v in pl.values())
            ac = sum(v['arrived_contact'] for v in pl.values())
            na = sum(v['not_arrived'] - v.get('not_arrived_solved', 0) for v in pl.values())
            cs = sum(v['contact_self'] for v in pl.values())
            co = sum(v['contact_obstacle'] for v in pl.values())
            rows.append([esc(run_name(r)), f'{f(pct(ok, n))}', f'{f(pct(ac, n))}', f'{f(pct(na, n))}', f'{f(pct(cs, max(cs + co, 1)))}', f(r['train_min'], 0), f'<code>{r["id"]}</code>', f'<code>{r["eval"]}</code>'])
        A('<h2 id="failures">How the runs fail</h2>')
        A('<p>A run fails either because the planner reaches the goal on a path that touches something, or because it does not reach the goal within its 600 steps (the final jump to the goal then collides).</p>')
        A('<div class="sumtbl">' + table(['method', 'solved %', 'arrived on a path in contact %', 'did not arrive %', 'first contact is self-contact, % of contacts', 'training min', 'field run', 'evaluation run'], rows, left=1) + '</div>')
        A('<p class="envnote">This table is this study’s own scoring (the scorer’s rule on this study’s copy of the geometry; it agrees with the cabinet evaluator to within a few runs).</p>')
    sp = OUT / f'stalls_{(lbs_runs[0]["id"][-8:] if lbs_runs else "")}.json'
    base_lbs = next((r for r in lbs_runs if r['label'] == 'lbs'), None)
    sp = OUT / f'stalls_{base_lbs["id"][-8:]}.json' if base_lbs else None
    if sp is not None and sp.exists():
        S = json.loads(sp.read_text())
        A('<h2 id="stops">Why the first LBS run stops short</h2>')
        if all(k in S for k in ('goal', 'ends', 'planner_variants', 'dock', 'block')):
            G, En, PV, Dk, Bk = S['goal'], S['ends'], S['planner_variants'], S['dock'], S['block']
            mm = lambda v: f'{v:.0f}'  # noqa: E731
            lab = lambda b: (f'{mm(b["lo_mm"]) if b["lo"] > 0 else "5"}–{mm(b["hi_mm"])} mm' if b['hi'] < 50 else 'over 50 mm')  # noqa: E731
            C['goalclear'] = hbar('Plans that reach the goal, by how close the goal is to an obstacle', 'First LBS run, planning seed 0, 600 queries. Clearance of the goal (or start) configuration; the speed model slows the arm below 50 mm.',
                                  [('by clearance of the goal', [(f'{lab(b)} · {b["n"]} queries', pct(b['arrived'], max(b['n'], 1)), 4, '') for b in G['by_goal_clearance']]),
                                   ('by clearance of the start (control)', [(f'{lab(b)} · {b["n"]} queries', pct(b['arrived'], max(b['n'], 1)), 5, '') for b in G['by_start_clearance']])],
                                  value_name='reached the goal')
            nfail = Dk['by_arc_cells']['0.5']['default plan did not arrive']['count']
            step_cells = S['planner']['step'] * 18.2
            C['dock'] = line('Started on a known collision-free path near the goal', f'The {nfail} queries whose plan did not arrive, restarted on the suite’s certified path a given distance before the goal. Same field and planner.',
                             [('queries whose plan did not arrive', 4, [(float(k) / step_cells, pct(v['default plan did not arrive']['solved'], nfail)) for k, v in Dk['by_arc_cells'].items()], None),
                              ('queries whose plan arrived (control)', 3, [(float(k) / step_cells, pct(v['default plan arrived']['solved'], v['default plan arrived']['count'])) for k, v in Dk['by_arc_cells'].items()], None)],
                             xlabel='distance before the goal along the certified path, planner steps', ylabel='solved %', xticks=[0, 16, 32, 48, 64], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, xmin=0, xmax=68,
                             digits=1, xname='steps before the goal', height=300)
            C['planner'] = hbar('Changing the planner does not help', 'First LBS run, planning seed 0, queries solved of 600. One setting changed at a time.',
                                [('', [(k.replace(' x', ' × ').replace('×  ', '× ') if k != 'default' else 'as evaluated (50 samples, horizon 5, 600 steps)', pct(v['success'], 600), 4 if k == 'default' else 5, '') for k, v in PV.items()])], value_name='solved')
            te = G['traversal_endpoints']
            mc = [k for k in te if float(k) > 0][0]
            g0 = G['by_goal_clearance'][0]
            gl = G['by_goal_clearance'][-1]
            d05 = Dk['by_arc_cells']['0.5']['default plan did not arrive']
            pv = [v['success'] for k, v in PV.items() if 'step x2' not in k]
            A(f'<p>About a third of the plans of the first LBS run do not reach the goal ({En["count"]} of 600 on planning seed 0). The cause is the field near goals that lie close to an obstacle, not the planner:</p>')
            A(f'<ul><li><b>It depends on the goal’s clearance.</b> Plans reach {f(pct(gl["arrived"], gl["n"]))} % of the goals that are more than 50 mm from every obstacle and {f(pct(g0["arrived"], g0["n"]))} % of the goals within {mm(g0["hi_mm"])} mm. '
              f'The clearance of the start makes no difference. Half of the test goals have the hand inside the cupboard, at a median clearance of {mm(1000 * G["test"]["goal_clearance_in_cupboard_q"][1] * G["cell_m"])} mm.</li>'
              f'<li><b>The plans stop with a panel in the way.</b> The median stop is {f(100 * En["remaining_over_line"][2], 0)} % of the start-to-goal joint distance short of the goal ({f(Bk["joint_move_rad_q"][2], 2)} rad), '
              f'{mm(1000 * En["clearance_cells"][2] * G["cell_m"])} mm from an obstacle, with the hand still {mm(Bk["grasp_point_to_goal_mm_q"][2] / 10)} cm from its goal position. '
              f'The straight move from there to the goal passes through the cupboard for {Bk["obstacle_blocks"]} of {Bk["count"]}.</li>'
              f'<li><b>The planner is not the limit.</b> Three times the steps, eight times the samples, a longer horizon, another temperature or rejecting rollouts in contact solve between {min(pv)} and {max(pv)} of 600; a doubled step size solves fewer.</li>'
              f'<li><b>The approach side is wrong.</b> Restarted on the suite’s certified path {f(0.5 / step_cells, 0)} planner steps before the goal, the same planner and field solve {d05["solved"]} of these {d05["count"]} queries. '
              f'So the field can guide the last stretch, but it leads the arm to the goal’s neighbourhood from a side that has no collision-free way in.</li>'
              f'<li><b>The field is close to plain distance near these goals.</b> At the stop the field’s time to the goal is {f(Dk["stop"]["T_median_not_arrived"], 3)}; on the certified path at the same distance from the goal it is '
              f'{f(Dk["by_arc_cells"]["0.5"]["default plan did not arrive"]["T_witness_point_median"], 3)}. The dead end looks slightly closer than the way in.</li></ul>')
            cup = next((r for r in lbs_runs if r['label'] == 'lbscup'), None)
            cup_txt = ''
            if cup is not None:
                oc_, ob_ = cup['official']['ntrl_mppi'], base_lbs['official']['ntrl_mppi']
                rf = cup['refresh'][-10:]
                cup_txt = (f' <b>Giving it such goals did not help:</b> with half of the endpoints drawn with the hand inside the cupboard, down to the 5 mm contact limit, the run solves {oc_["success"]:,} of {oc_["n"]:,} '
                           f'(first run {ob_["success"]:,}). Its traversals reach their goal {f(pct(sum(x["reached"] for x in rf), sum(x["traversals"] for x in rf)), 0)} % of the time and stall otherwise, '
                           'so the segment loss, which learns from traversals that get through, has little to learn from where it is needed.')
            A(f'<p><b>The segment loss does not correct this.</b> Its traversals ran between uniformly drawn free configurations at least {mm(1000 * (float(mc) + G["contact_cells"]) * G["cell_m"])} mm from every obstacle; '
              f'{f(100 * te[mc]["in_cupboard"], 1)} % of them have the hand in the cupboard, against {f(100 * G["test"]["goals_in_cupboard"], 0)} % of the test goals.{cup_txt}</p>')
            A(f'<div class="grid2">{chart("goalclear")}{chart("dock")}</div>')
            A(chart('planner'))
            fdone = [r for r in lbs_runs if max(r['freq']) > 1 + 1e-9]
            if not fdone:
                A('<p>Open: whether a finer joint Fourier frequency lets the field hold the thin panels. The runs at × 2 and × 4 are training. More samples in the cupboard, for the pairs or for the segment endpoints, did not change the result (see the knobs).</p>')
    rw = 'real_world_6dof'
    rw_pend = [q for q in D['pending'] if q['label'].startswith('rw_')]
    if rw in D.get('scenes', {}) and (real or rw_pend):
        lf = D['scenes'][rw]['line_feasible']
        nl = sum(lf.values())
        A('<h2 id="real">Real-world cupboard (real_world_6dof)</h2>')
        A('<p>The measured cupboard: 1.82 m tall, 0.92 m wide, 0.56 m deep, three zones 0.53 m tall behind an open front, standing on the floor. One UR5e with its gripper on a pedestal, base 0.70 m above the floor and 0.45 m in front of the cupboard. '
          f'600 new certified queries were generated with the cabinet recipe (six suites of 100; the certificate is a straight line or an RRT-Connect path that the release scorer accepts); a straight line solves {nl} of them. '
          'Training and planner settings are those of the first LBS run of the cabinet. No earlier ntrl-demo or P-NTFields run exists for this scene, so the control is ntrl-demo trained here on the same data.</p>')
        A('<p class="envnote">One change to the measured scene: the pedestal’s top is lowered by 4 cm. In the measured scene it touches the arm’s base, and the release scorer, which also checks the arm’s two fixed base spheres, then rejects every configuration.</p>')
        if real:
            rows_a = [('straight line (control)', pct(nl, 600), 6, 'no planner')] + [(run_name(r), pct(r['official']['ntrl_mppi']['success'], r['official']['ntrl_mppi']['n']), slot(r), f'field {r["id"][-8:]}, evaluation {r["eval"][-8:]}') for r in real]
            rows_d = [('straight line (control)', 0.0, 6, 'by definition')] + [(run_name(r), pct(*r['official']['ntrl_mppi']['detour']), slot(r), '') for r in real]
            nd = real[0]['official']['ntrl_mppi']['detour'][1]
            C['real_all'] = hbar('Real-world cupboard: queries solved', 'The cabinet evaluator: ntrl-demo’s MPPI, planning seeds 0, 1 and 2, release scorer. Training seed 87358.',
                                 [('all 600 queries (1,800 runs)', rows_a), (f'the {nd // 3} queries that need a detour ({nd:,} runs)', rows_d)], value_name='solved')
            C['real_tasks'] = hbar('Real-world cupboard: queries solved by suite', 'Each suite has 100 queries; the bars are the percentage solved over the three planning seeds.',
                                   [(TASK[t], [('straight line (control)', float(lf[t]), 6, '')] + [(run_name(r), r['official']['ntrl_mppi']['per_task'][t] / 3, slot(r), '') for r in real]) for t in D['tasks']], value_name='solved')
            rl = next((r for r in real if all(r['arm'].values())), None)
            rc = next((r for r in real if not any(r['arm'].values())), None)
            if rl and rc:
                a_, c_ = rl['official']['ntrl_mppi'], rc['official']['ntrl_mppi']
                A(f'<p><b>LBS solves {f(pct(a_["success"], a_["n"]))} % of the runs</b> ({a_["success"]:,} of {a_["n"]:,}), ntrl-demo {f(pct(c_["success"], c_["n"]))} % ({c_["success"]:,}), a straight line {f(pct(nl, 600))} %. '
                  f'On the queries that need a detour: LBS {f(pct(*a_["detour"]))} %, ntrl-demo {f(pct(*c_["detour"]))} %. '
                  f'Zone → another zone: LBS {a_["per_task"]["compartment_to_compartment"]} of 300, ntrl-demo {c_["per_task"]["compartment_to_compartment"]} of 300.</p>')
                ro = next((r for r in real if r['arm']['L'] and not r['arm']['B'] and not r['arm']['S']), None)
                if ro:
                    o_ = ro['official']['ntrl_mppi']
                    A(f'<p><b>The lift alone</b> (no Bellman update, no segment loss) solves {f(pct(o_["success"], o_["n"]))} % ({o_["success"]:,}; detour runs {f(pct(*o_["detour"]))} %), '
                      + ('level with all three together, as on the cabinet.' if abs(o_['success'] - a_['success']) < 0.02 * a_['n'] else 'against all three together above.') + '</p>')
            elif rc:
                c_ = rc['official']['ntrl_mppi']
                A(f'<p>So far only the control is scored: ntrl-demo solves {f(pct(c_["success"], c_["n"]))} % of the runs ({c_["success"]:,} of {c_["n"]:,}), {f(pct(*c_["detour"]))} % of the detour runs; a straight line solves {f(pct(nl, 600))} %. The LBS run is training.</p>')
            A(chart('real_all'))
            A(chart('real_tasks'))
            rows = []
            for r in real:
                o = r['official']['ntrl_mppi']
                rows.append([esc(run_name(r)), f'{o["success"]:,} ({f(pct(o["success"], o["n"]))} %)', f'{o["detour"][0]} ({f(pct(*o["detour"]))} %)', f'{o["per_task"]["compartment_to_compartment"]} / 300',
                             f'{r["official"]["native_rk4"]["success"]} / 600', f(r['train_min'], 0), f'<code>{r["id"]}</code>', f'<code>{r["eval"]}</code>'])
            A(details('Runs on the real-world cupboard', table(['method', 'solved of 1,800 runs', 'detour runs solved', 'zone → another zone', 'RK4 descent', 'training min', 'field run', 'evaluation run'], rows, left=1)))
        else:
            A('<p>The ntrl-demo control and the LBS run for this scene are ' + ('training' if any(q.get('epoch') is not None for q in rw_pend) else 'in the queue') + '.</p>')
    r12 = 'real_world_12dof'
    p12 = [q for q in D['pending'] if q['label'].startswith('rw12b')]
    eu = D.get('euclid', {}).get(r12)
    if r12 in D.get('scenes', {}) and (twelve or p12 or short12):
        lf = D['scenes'][r12]['line_feasible']
        nl = sum(lf.values())
        A('<h2 id="real12">Real-world cupboard, two arms (real_world_12dof)</h2>')
        A('<p>The same measured cupboard with two UR5e arms side by side, 0.60 m apart, bases 0.60 m above the floor and 0.25 m in front of the cupboard. A configuration is 12 joints; the lift makes it 13 inputs. '
          f'600 certified queries in the six suites, the regions holding for both grasp points; a straight line solves {nl} of them.</p>')
        A('<p class="envnote">What differs from the one-arm scenes. (1) The release scorer and evaluator are for one arm, so these numbers come from this study’s own scorer: each arm by the release rule '
          '(5 mm from every obstacle, 2 mm self clearance), the two arms at least 5 mm apart, paths checked every 0.001 rad per joint. The same rule certified the suites; its distances agree with the separately written scene model '
          'to 0.3 µm. (2) The witness of a query is a straight line, an RRT-Connect path in all 12 joints, or one arm moving after the other; the suites keep only queries with such a witness (see the setup of this scene below). '
          '(3) Both pedestals are lowered by 4 cm, as in the one-arm scene.</p>')
        own = lambda r: r['own']  # noqa: E731
        if eu:
            A(f'<p><b>Planner control.</b> The same planner on the Euclidean field T = |x − g| (no network) arrives on every query and solves {eu["success"][0]:,} of {eu["success"][1]:,} runs ({f(pct(*eu["success"]))} %): '
              f'{eu["line"][0]} of the {eu["line"][1]} straight-line runs and {eu["detour"][0]} of the {eu["detour"][1]:,} detour runs. So the planner settings carry over to 12 joints, and a field has to beat {f(pct(*eu["success"]))} %, not 0. '
              + 'More samples or more steps change little (' + ', '.join(f'{q["steps"]:,} steps and {q["samples"]} samples: {f(pct(*q["success"]))} %' for q in eu['settings'][1:]) + f'). Run <code>{eu["id"]}</code>.</p>')
        if short12:
            r0 = short12[0]
            n0 = sum(v['runs'] for v in own(r0)['plan'].values())
            A(details('A first control that does not count: training pairs too short', f'<p>The first training set drew a pair as in the one-arm scenes: a start near an obstacle, a step of uniform length up to 0.71 in a random direction, '
              'kept if the far end is not in collision. With two arms almost every long step ends in collision, so the kept pairs were short: median length 0.06, and 3.6 % as long as the median test query (0.46); '
              'in the one-arm real-world scene 22 % are. The long ones also ended outside the task box, where no query is. '
              f'ntrl-demo trained on it solved {own(r0)["success"][0]} of {own(r0)["success"][1]:,} runs, fewer than the Euclidean field, and {f(pct(sum(v["not_arrived"] for v in own(r0)["plan"].values()), n0), 0)} % of the runs never reached the goal '
              f'(run <code>{r0["id"]}</code>). The runs of this section use a second training set: the same rule, but the kept pairs fill 20 equal length bins of 0 to 0.71 equally, and in every bin half of the far ends have both grasp points in the task box '
              '(56 % in total, the share the one-arm scene has).</p>'))
        main12 = [r for r in twelve if not r['label'].startswith('rw12b_lift_')]
        knob12 = [r for r in twelve if r['label'].startswith('rw12b_lift_')]
        k12 = lambda r: ''.join(k for k in 'LBS' if r['arm'][k])  # noqa: E731
        spread = lambda r: '--pair-data' in r.get('extra', '')  # noqa: E731
        n12 = lambda r: {'': 'ntrl-demo (control)', 'L': 'lift only', 'LBS': 'LBS (lift, Bellman, segment)', 'LS': 'lift + segment (no Bellman)', 'LB': 'lift + Bellman (no segment)'}.get(k12(r), k12(r)) \
            + (', traversal lengths spread' if spread(r) else '') + (f', frequency × {r["freq"][0]:g}' if abs(r['freq'][0] - 1) > 1e-9 else '')  # noqa: E731
        allmain12 = main12
        main12 = [r for r in allmain12 if abs(r['freq'][0] - 1) < 1e-9]
        if twelve:
            euc = [('Euclidean field, same planner (control)', pct(*eu['success']), 6, f'run {eu["id"][-8:]}')] if eu else []
            eud = [('Euclidean field, same planner (control)', pct(*eu['detour']), 6, '')] if eu else []
            rows_a = [('straight line (control)', pct(nl, 600), 6, 'no planner')] + euc + [(n12(r), pct(*own(r)['success']), slot(r), f'field {r["id"][-8:]}') for r in main12]
            rows_d = [('straight line (control)', 0.0, 6, 'by definition')] + eud + [(n12(r), pct(*own(r)['detour']), slot(r), '') for r in main12]
            nd = own(twelve[0])['detour'][1]
            C['real12_all'] = hbar('Two arms, 12 joints: queries solved', 'ntrl-demo’s MPPI, planning seeds 0, 1 and 2, this study’s scorer. Training seed 87358.',
                                   [('all 600 queries (1,800 runs)', rows_a), (f'the {nd // 3} queries that need a detour ({nd:,} runs)', rows_d)], value_name='solved')
            C['real12_tasks'] = hbar('Two arms: queries solved by suite', 'Each suite has 100 queries; the bars are the percentage solved over the three planning seeds.',
                                     [(TASK[t], [('straight line (control)', float(lf[t]), 6, '')] + ([('Euclidean field, same planner (control)', eu['by_task'][t] / 3, 6, '')] if eu else []) + [(n12(r), own(r)['by_task'][t] / 3, slot(r), '') for r in main12]) for t in D['tasks']], value_name='solved')
            tc = next((r for r in main12 if k12(r) == ''), None)
            to = next((r for r in main12 if k12(r) == 'L'), None)
            tl = next((r for r in main12 if k12(r) == 'LBS' and not spread(r)), None)
            pd = [r for r in main12 if spread(r)]
            half = [r for r in twelve if abs(r['freq'][0] - 0.5) < 1e-9 and abs(r['lift']['gain'] - 1) < 1e-9]
            sv = lambda r: f'{f(pct(*own(r)["success"]))} % of the runs ({own(r)["success"][0]:,} of {own(r)["success"][1]:,}; {f(pct(*own(r)["detour"]))} % of the detour runs)'  # noqa: E731
            if tc:
                A(f'<p><b>The control.</b> ntrl-demo solves {sv(tc)}; a straight line solves {f(pct(nl, 600))} %' + (f', the Euclidean field with the same planner {f(pct(*eu["success"]))} %' if eu else '')
                  + '. The control is below both: its field holds a third of the runs at the start and sends most of the others into contact.</p>')
            if to:
                A(f'<p><b>The lift alone</b> (no Bellman update, no segment loss) solves {sv(to)}: above every control, but far from the 62 % it reaches with one arm. Most of its runs stall before the goal, and the longer the query the more often: '
                  '42 % of the shortest fifth of the queries arrive (joint distance 0.17 to 0.40 of the joint range), 1 % of the longest fifth (0.55 to 0.80).</p>')
            if knob12 and to:
                kn_ = lambda r: (f'h range {r["lift"]["h_range"]:.2f}' if abs(r['lift']['gain'] - 1) > 1e-9 else f'Fourier frequency × {r["freq"][0]:g}')  # noqa: E731
                kb = max(knob12, key=lambda r: own(r)['success'][0])
                if own(kb)['success'][0] > 1.5 * own(to)['success'][0]:
                    A(f'<p><b>One knob changes this: {kn_(kb)}.</b> The lift alone then solves {sv(kb)}, against {f(pct(*own(to)["success"]))} % at the setting of the one-arm scenes. '
                      + ('Other knob runs: ' + ', '.join(f'{kn_(r)} {f(pct(*own(r)["success"]))} %' for r in knob12 if r is not kb) + '. ' if len(knob12) > 1 else 'The other knob runs (frequency × 0.25, three ranges of h) are in the queue. ')
                      + 'One training seed.</p>')
            if tl:
                rr = [q['reached'] for q in tl['refresh']]
                A(f'<p><b>LBS</b> (lift, Bellman, segment) solves {sv(tl)}' + (f', level with the lift alone ({own(to)["success"][0]:,}).' if to and abs(pct(*own(tl)['success']) - pct(*own(to)['success'])) < 2 else '.')
                  + (f' In this run the segment loss had almost nothing to learn from: of the {tl["refresh"][0]["traversals"]:,} training traversals of a refresh, between {min(rr)} and {max(rr)} reached their goal. '
                     'The traversal pairs are random free configurations at most 0.85 apart, as in the one-arm scenes, and with 12 joints such pairs are almost all long: 90 % are longer than 0.65, '
                     'and 95 % of the test queries are shorter than that. A field that arrives only on short queries never completes them, so the segment loss cannot start.' if rr else '') + '</p>')
                if pd:
                    A('<p><b>Traversal pairs of every length.</b> Pairs taken from the training pairs instead (both ends free, lengths spread evenly over 0 to 0.71): ' + '; '.join(f'{n12(r).split(",")[0]} {sv(r)}' for r in pd) + '.</p>')
                else:
                    A('<p>Runs with traversal pairs of every length (taken from the training pairs, lengths spread evenly over 0 to 0.71) are '
                      + ('training' if any(q.get('epoch') is not None for q in p12) else 'in the queue') + '; they are the fair test of the segment loss on this scene.</p>')
            if len(half) > 1:
                hk = {(k12(r), spread(r)): r for r in half}
                hn = lambda r: n12(r).split(', frequency')[0]  # noqa: E731
                txt = '<p><b>The other parts at half the frequency.</b> ' + '; '.join(f'{hn(r)} {sv(r)}' for r in half) + '. '
                if ('L', False) in hk and ('LS', True) in hk:
                    a_, b_ = hk[('L', False)], hk[('LS', True)]
                    rr = [q['reached'] for q in b_['refresh']]
                    txt += (f'The segment loss was active in its run (between {min(rr):,} and {max(rr):,} of the {b_["refresh"][0]["traversals"]:,} training traversals of a refresh reached their goal) and it changes nothing measurable on top of the lift: '
                            f'{own(b_)["success"][0]} against {own(a_)["success"][0]} runs solved. ')
                if ('LBS', True) in hk and ('L', False) in hk:
                    c_ = lambda r: sum(v['arrived_contact'] for v in own(r)['plan'].values())  # noqa: E731
                    txt += (f'Full LBS is below the lift alone, {own(hk[("LBS", True)])["success"][0]} against {own(hk[("L", False)])["success"][0]} runs, and more of its paths end in contact ({c_(hk[("LBS", True)])} against {c_(hk[("L", False)])}); '
                            'the second seed, next, shows how much a seed moves these numbers. ')
                if ('', False) in hk and ('L', False) in hk:
                    txt += f'The lift is still what carries the result: without it, half the frequency gives {f(pct(*own(hk[("", False)])["success"]))} %. '
                hp = [q['label'] for q in D['pending'] if q['label'] in ('rw12b_ntrl_f05', 'rw12b_lbs_pd_f05')]
                if hp:
                    txt += 'Still training: ' + ' and '.join({'rw12b_ntrl_f05': 'the ntrl-demo control at half the frequency', 'rw12b_lbs_pd_f05': 'full LBS at half the frequency'}[q] for q in hp) + '. '
                s2 = {(k12(r), spread(r)): r for r in D['runs'] if r.get('scene') == 'real_world_12dof' and r['seed'] != SEED and r['label'].startswith('rw12b') and abs(r['freq'][0] - 0.5) < 1e-9 and 'own' in r}
                if s2:
                    txt += (f'A second training seed ({next(iter(s2.values()))["seed"]}): ' + ', '.join(f'{n12(r).split(", traversal")[0].split(", frequency")[0]} {own(r)["success"][0]} (first seed {own(hk[k])["success"][0]})' for k, r in s2.items() if k in hk)
                            + '. The two seeds differ by 30 to 50 runs, so lift only and lift + segment are level' + ('' if ('LBS', True) in s2 else '; the second seed of full LBS is training') + '. '
                            + (f'Full LBS is below the lift alone on both seeds, by {own(hk[("L", False)])["success"][0] - own(hk[("LBS", True)])["success"][0]} and {own(s2[("L", False)])["success"][0] - own(s2[("LBS", True)])["success"][0]} runs: '
                               'with two arms the Bellman update costs planning at this frequency. ' if all(k in s2 and k in hk for k in (('LBS', True), ('L', False))) else ''))
                q12 = {k12(r): r for r in twelve if abs(r['freq'][0] - 0.25) < 1e-9}
                if 'LBS' in q12 and 'L' in q12:
                    txt += f'At a quarter of the frequency full LBS solves {own(q12["LBS"])["success"][0]} runs and the lift alone {own(q12["L"])["success"][0]}: lowering the frequency further does not close the gap. '
                if ('', False) in hk:
                    pl_ = own(hk[('', False)])['plan']
                    txt += (f'The control arrives almost always but {f(pct(sum(v["arrived_contact"] for v in pl_.values()), sum(v["runs"] for v in pl_.values())), 0)} % of its paths are in contact: its field does not see the obstacles. ')
                A(txt + 'In the runs with the lift at half the frequency almost no path ends in contact; the unsolved runs stop before the goal.' + ('' if s2 else ' One training seed.') + '</p>')
                order = [('', False), ('L', False), ('LS', True), ('LBS', True)]
                g1 = {(k12(r), spread(r)): r for r in main12}
                if ('LBS', True) not in g1 and ('LBS', False) in g1:
                    g1[('LBS', True)] = g1[('LBS', False)]
                C['real12_half'] = hbar('Two arms: every part at frequency × 1 and × 0.5', 'Queries solved of the 1,800 runs, this study’s scorer. Lift + segment and LBS use traversal pairs of every length where that run exists.',
                                        [(nm, [(n12(r).split(', frequency')[0], pct(*own(r)['success']), slot(r), f'field {r["id"][-8:]}') for r in (g.get(k) for k in order) if r])
                                         for nm, g in (('frequency × 1', g1), ('frequency × 0.5', hk))], value_name='solved')
            A(chart('real12_all'))
            A(chart('real12_tasks'))
            if len(half) > 1:
                A(chart('real12_half'))
            if knob12 and to:
                kn = lambda r: (f'h range {r["lift"]["h_range"]:.2f}' if abs(r['lift']['gain'] - 1) > 1e-9 else f'frequency × {r["freq"][0]:g}')  # noqa: E731
                C['real12_knobs'] = hbar('Two arms, lift only: range of h and Fourier frequency', f'Queries solved of the 1,800 runs; one knob changed from the lift-only run (h range {to["lift"]["h_range"]:.2f}, frequency × 1).',
                                         [('lift only', [(f'h range {to["lift"]["h_range"]:.2f}, frequency × 1', pct(*own(to)['success']), 4, '')] + [(kn(r), pct(*own(r)['success']), 5, '') for r in knob12])], value_name='solved')
                A(chart('real12_knobs'))
            rows = []
            for r in allmain12 + knob12:
                pl = r['own']['plan']
                n = sum(v['runs'] for v in pl.values())
                rows.append([esc(n12(r) + (', ' + kn(r) if r in knob12 and to else '')), f'{own(r)["success"][0]:,} ({f(pct(*own(r)["success"]))} %)', f'{own(r)["detour"][0]} ({f(pct(*own(r)["detour"]))} %)', f'{own(r)["by_task"]["compartment_to_compartment"]} / 300',
                             f(pct(sum(v['arrived_contact'] for v in pl.values()), n)), f(pct(sum(v['not_arrived'] - v.get('not_arrived_solved', 0) for v in pl.values()), n)), f(r['train_min'], 0), f'<code>{r["id"]}</code>'])
            A(details('Runs on the two-arm scene', table(['method', 'solved of 1,800 runs', 'detour runs solved', 'zone → another zone', 'arrived on a path in contact %', 'did not arrive %', 'training min', 'field run'], rows, left=1)))
        else:
            A('<p>The ntrl-demo control and the LBS run for this scene are ' + ('training' if any(q.get('epoch') is not None for q in p12) else 'in the queue') + '.</p>')
        g12 = OUT.parent / 'tests_real_world_12dof/generation.json'
        if g12.exists():
            G12 = json.loads(g12.read_text())
            rows = [[TASK[t], str(v['candidates']), str(v.get('certified', 0)), f(pct(v.get('certified', 0), v['candidates']), 0), str(v['selected_direct']),
                     ', '.join(f'{k.replace("rrtc_one_arm_at_a_time", "one arm at a time").replace("rrtc", "RRT-Connect, all joints").replace("direct", "straight line")} {n}' for k, n in v['selected_sources'].items())]
                    for t, v in G12['strata'].items()]
            A(details('How the two-arm suites were generated', '<p>Endpoints: one configuration per arm, valid for that arm alone, with its grasp point in the suite’s region; kept if the two arms are clear of each other. '
              f'Pairs: joint distance 0.02 to {f(G12["parameters"]["max_cspace_l2"], 2)} rad (4 rad per arm, as the one-arm suites). A candidate enters the suite only with a certified witness, so the suites lean towards queries '
              'that these witness planners can solve.</p>' + table(['suite', 'candidates', 'certified', 'certified %', 'straight line among the 100', 'witnesses of the 100'], rows, left=1)))
    CL = D.get('close', {})
    if CL:
        A('<h2 id="close">Close pairs: predicted against true travel time</h2>')
        A('<p>A direct check of the values, not of planning. A pair is a valid configuration and the same configuration moved by a short step in a random direction, kept if the straight step stays collision-free; '
          '5,000 pairs per step length. The true time is the travel time of that straight step at the training speed, the step cost the Bellman update uses; for steps this short it is the true travel time or a tight upper bound. '
          'A step of 1 cell is 0.063 of the joint range with one arm and 0.072 with two; the median test query is 0.30 and 0.46.</p>')
        cn = {'ntrl': ('ntrl-demo (control)', 3, None), 'lift': ('lift only', 1, None), 'lbs': ('LBS (lift, Bellman, segment)', 4, None), 'lift_segment': ('lift + segment', 2, '6 3'), 'lift_bellman': ('lift + Bellman', 5, '2 3')}
        for sn, ttl in (('real_world_6dof', 'One arm, real-world cupboard'), ('real_world_12dof', 'Two arms, real-world cupboard'), ('cabinet', 'One arm, cabinet, with the ablations')):
            if sn not in CL:
                continue
            ser = [(cn[k][0], cn[k][1], [(float(c), v['ratio'][1]) for c, v in sorted(CL[sn][k]['rows'].items(), key=lambda t: float(t[0]))], cn[k][2]) for k in cn if k in CL[sn]]
            top = max(y for _, _, pts, _ in ser for _, y in pts)
            C['close_' + sn] = line(ttl + ': median predicted ÷ true time', 'By step length. 1 = the field predicts the true travel time; above 1 it overestimates.', ser, xlabel='step length, cells', ylabel='predicted ÷ true',
                                    xticks=[0.25, 0.5, 1, 2, 4], yticks=[0, 1, 2, 3, 4, 5] if top > 2 else [0, 0.5, 1, 1.5, 2], ymin=0, ymax=5 if top > 2 else 2, xmin=0.25, xmax=4, logx=True, digits=2, xname='cells', height=280,
                                    refs=[(1.0, 'true time')])
        six, tw, cb = CL.get('real_world_6dof', {}), CL.get('real_world_12dof', {}), CL.get('cabinet', {})
        r1 = lambda c, k: c[k]['rows']['1.0']  # noqa: E731
        if six and tw and all(k in six and k in tw for k in ('ntrl', 'lift', 'lbs')):
            A(f'<p><b>One arm.</b> The lift-only field is close to the truth: median {f(r1(six, "lift")["ratio"][1], 2)} at 1 cell, {f(100 * r1(six, "lift")["within"], 0)} % of the pairs within 20 %. '
              f'ntrl-demo is {f(r1(six, "ntrl")["ratio"][1], 2)}; LBS overestimates, {f(r1(six, "lbs")["ratio"][1], 2)}, with {f(100 * r1(six, "lbs")["within"], 0)} % within 20 %. '
              f'<b>Two arms.</b> No field is close: within 20 % are {f(100 * r1(tw, "ntrl")["within"], 0)} % (ntrl-demo), {f(100 * r1(tw, "lift")["within"], 0)} % (lift only) and {f(100 * r1(tw, "lbs")["within"], 0)} % (LBS, median {f(r1(tw, "lbs")["ratio"][1], 2)}). '
              f'The medians of ntrl-demo and lift only hide two opposite errors: steps in open space are overestimated ({f(r1(tw, "ntrl")["groups"]["full_speed"]["ratio"], 2)} and {f(r1(tw, "lift")["groups"]["full_speed"]["ratio"], 2)}), '
              f'steps next to obstacles underestimated ({f(r1(tw, "ntrl")["groups"]["slow"]["ratio"], 2)} and {f(r1(tw, "lift")["groups"]["slow"]["ratio"], 2)}), and with two arms '
              f'{f(pct(r1(tw, "lift")["groups"]["slow"]["n"], r1(tw, "lift")["n"]), 0)} % of the steps are next to obstacles (one arm: {f(pct(r1(six, "lift")["groups"]["slow"]["n"], r1(six, "lift")["n"]), 0)} %).</p>')
        if cb and all(k in cb for k in ('lift', 'lift_segment', 'lift_bellman', 'lbs')):
            A(f'<p><b>The overestimate of LBS comes with the Bellman update.</b> On the cabinet at 1 cell: lift only {f(r1(cb, "lift")["ratio"][1], 2)}, lift + segment {f(r1(cb, "lift_segment")["ratio"][1], 2)}, '
              f'lift + Bellman {f(r1(cb, "lift_bellman")["ratio"][1], 2)}, all three {f(r1(cb, "lbs")["ratio"][1], 2)}. The segment loss with min leaves adds no bias of its own; it also does not remove the Bellman update’s.</p>')
        for sn in ('real_world_6dof', 'real_world_12dof', 'cabinet'):
            A(ch('close_' + sn))
        kf = [(k, n) for k, n in (('f025', 'frequency × 0.25'), ('f05', 'frequency × 0.5'), ('lbs', 'frequency × 1 (first LBS run)'), ('f2', 'frequency × 2'), ('f4', 'frequency × 4')) if k in cb]
        kg = [(k, n) for k, n in (('g04', 'h range 0.05'), ('lbs', 'h range 0.12 (first LBS run)'), ('g2', 'h range 0.29'), ('g6', 'h range 0.71')) if k in cb]
        if len(kf) > 2:
            byl = {r['label']: r for r in every}
            lab = dict(f025='lbsf025', f05='lbsf05', lbs='lbs', f2='lbsf2', f4='lbsf4', g04='lbsg04', g2='lbsg2', g6='lbsg6')
            sol = lambda k: (f'{f(pct(byl[lab[k]]["official"]["ntrl_mppi"]["success"], 1800))} % of the queries solved' if lab[k] in byl else '')  # noqa: E731
            C['close_knobs'] = hbar('Cabinet, LBS: predicted ÷ true time by knob', 'Median over 5,000 steps of 1 cell; 1 = true. One knob changed at a time from the first LBS run.',
                                    [('joint Fourier frequency', [(n, r1(cb, k)['ratio'][1], 4 if k == 'lbs' else 5, sol(k)) for k, n in kf]), ('range of h', [(n, r1(cb, k)['ratio'][1], 4 if k == 'lbs' else 5, sol(k)) for k, n in kg])],
                                    xmax=4, ticks=(0, 1, 2, 3, 4), digits=2, suffix='', value_name='predicted ÷ true')
            A('<p><b>The overestimate follows the Fourier frequency.</b> LBS on the cabinet: ' + ', '.join(f'{n.split(" (")[0]} {f(r1(cb, k)["ratio"][1], 2)}' for k, n in kf)
              + '. The two runs at a lower frequency are close to the truth and plan as well as the first run; the two at a higher frequency overestimate two- and four-fold and are the ones whose planning collapsed. '
              'The range of h changes little (' + ', '.join(f'{f(r1(cb, k)["ratio"][1], 2)}' for k, n in kg) + ').</p>')
            A(ch('close_knobs'))
        h12 = [(k, n, sl) for k, n, sl in (('ntrlf05', 'ntrl-demo (control)', 3), ('liftf05', 'lift only', 1), ('lsf05', 'lift + segment', 2), ('lbsf05', 'LBS (lift, Bellman, segment)', 4)) if k in tw]
        if len(h12) > 2 and all(k in tw for k in ('ntrl', 'lift', 'lbs')):
            g_ = lambda k, g: r1(tw, k)['groups'][g]['ratio']  # noqa: E731
            nt = lambda k: f'open space {f(g_(k, "full_speed"), 2)}, next to obstacles {f(g_(k, "slow"), 2)}'  # noqa: E731
            C['close_half12'] = hbar('Two arms: predicted ÷ true time at frequency × 1 and × 0.5', 'Median over 5,000 steps of 1 cell; 1 = true. Hover a bar for the medians in open space and next to obstacles.',
                                     [('frequency × 1', [(cn[k][0], r1(tw, k)['ratio'][1], cn[k][1], nt(k)) for k in ('ntrl', 'lift', 'lbs')]), ('frequency × 0.5', [(n, r1(tw, k)['ratio'][1], sl, nt(k)) for k, n, sl in h12])],
                                     xmax=4, ticks=(0, 1, 2, 3, 4), digits=2, suffix='', value_name='predicted ÷ true')
            if 'lbsf05' in tw and 'liftf05' in tw:
                A(f'<p><b>With two arms half the frequency does not remove it.</b> LBS at 1 cell goes from {f(r1(tw, "lbs")["ratio"][1], 2)} to {f(r1(tw, "lbsf05")["ratio"][1], 2)}; in open space it still predicts {f(g_("lbsf05", "full_speed"), 2)} times the true time. '
                  f'The lift-only field at half the frequency has a median of {f(r1(tw, "liftf05")["ratio"][1], 2)}, but that is again two opposite errors: {f(g_("liftf05", "full_speed"), 2)} in open space and {f(g_("liftf05", "slow"), 2)} next to obstacles'
                  + (f'; the segment loss leaves it there ({f(r1(tw, "lsf05")["ratio"][1], 2)})' if 'lsf05' in tw else '') + '. '
                  + (f'At a quarter of the frequency LBS is {f(r1(tw, "lbsf025")["ratio"][1], 2)} ({f(g_("lbsf025", "full_speed"), 2)} in open space). ' if 'lbsf025' in tw else '')
                  + f'Test run <code>{tw["lbsf05"]["run"]}</code>.</p>')
            A(ch('close_half12'))
        rows = [[{'real_world_6dof': 'one arm, real-world', 'real_world_12dof': 'two arms', 'cabinet': 'cabinet'}[sn], cn.get(k, (k,))[0], c] + [f'{f(v["rows"][c]["ratio"][0], 2)} · {f(v["rows"][c]["ratio"][1], 2)} · {f(v["rows"][c]["ratio"][2], 2)}', f(100 * v['rows'][c]['within'], 0),
                 ' · '.join(f(v['rows'][c]['groups'][g]['ratio'], 2) if g in v['rows'][c]['groups'] else '–' for g in ('full_speed', 'slowed', 'slow')), f'<code>{v["field_run"]}</code>', f'<code>{v["run"]}</code>']
                for sn in ('real_world_6dof', 'real_world_12dof', 'cabinet') if sn in CL for k, v in CL[sn].items() if k in cn for c in sorted(v['rows'], key=float)]
        A(details('Close pairs: all numbers', table(['scene', 'field', 'step, cells', 'predicted ÷ true, 10 · 50 · 90 %', 'within 20 %, %', 'median by speed: full · slowed · slow', 'field run', 'test run'], rows, left=2)))
    if leafx and 'real_world_6dof' in CL:
        six = CL['real_world_6dof']
        base_ = next((r for r in real if all(r['arm'].values())), None)
        lo_ = next((r for r in real if r['arm']['L'] and not r['arm']['B'] and not r['arm']['S']), None)
        A('<h2 id="leaves">Segment-loss versions on the one-arm scene</h2>')
        A('<p>The question: can the segment loss remove the overestimate of LBS? The LBS runs above already use the main version of the midpoint-loss study: wall-avoiding drives, min leaves (a leaf is worth the smaller of the network’s time '
          'and the time the drive took over it), tree and sub-paths, weight 1, leaves of 1 cell. Each run below is full LBS on the one-arm real-world scene with one change.</p>')
        vn = {'rw_lbs_w3': ('min leaves, weight 3', 'w3'), 'rw_lbs_w10': ('min leaves, weight 10', 'w10'), 'rw_lbs_w30': ('min leaves, weight 30', 'w30'), 'rw_lbs_w100': ('min leaves, weight 100', 'w100'), 'rw_lbs_lpath': ('path leaves, weight 1', 'lpath'),
              'rw_lbs_lnet': ('net leaves, weight 1', 'lnet'), 'rw_lbs_bgoal': ('weight 1, Bellman also tries the goal step', 'bgoal'),
              'rw_lbs_lstep': ('true-step leaves, weight 1', 'lstep'), 'rw_lbs_f05': ('min leaves, weight 1, frequency × 0.5', 'lbsf05'), 'rw_lbs_f025': ('min leaves, weight 1, frequency × 0.25', 'f025')}
        lst = ([('LBS as above: min leaves, weight 1', base_, 'lbs', 4)] if base_ else []) + [(vn[r['label']][0], r, vn[r['label']][1], 5) for r in sorted((r for r in leafx if r['label'] in vn), key=lambda r: list(vn).index(r['label']))] + ([('lift only, for reference', lo_, 'lift', 1)] if lo_ else []) + [('lift only, frequency × 0.5', r, 'liftf05', 1) for r in every if r['label'] == 'rw_lift_f05']
        have = [(n, r, k, s) for n, r, k, s in lst if k in six]
        C['leaf_ratio'] = hbar('Segment-loss versions: predicted ÷ true time on close pairs', 'Median over 5,000 steps of 1 cell. 1 = true.', [('', [(n, six[k]['rows']['1.0']['ratio'][1], s, f'{f(100 * six[k]["rows"]["1.0"]["within"], 0)} % within 20 %') for n, r, k, s in have])],
                               xmax=2, ticks=(0, 0.5, 1, 1.5, 2), digits=2, suffix='', value_name='predicted ÷ true')
        C['leaf_solved'] = hbar('Segment-loss versions: queries solved', 'The cabinet evaluator, 1,800 runs.', [('', [(n, pct(r['official']['ntrl_mppi']['success'], r['official']['ntrl_mppi']['n']), s, f'field {r["id"][-8:]}') for n, r, k, s in lst])], value_name='solved')
        ws = [(n, r, k) for n, r, k, s in have if k in ('lbs', 'w3', 'w10', 'w30', 'w100')]
        if len(ws) > 1:
            A('<p><b>A heavier segment loss lowers the overestimate, slowly:</b> ' + ', '.join(f'weight {dict(lbs=1, w3=3, w10=10, w30=30, w100=100)[k]}: {f(six[k]["rows"]["1.0"]["ratio"][1], 2)}' for n, r, k in ws)
              + (f' (lift only: {f(six["lift"]["rows"]["1.0"]["ratio"][1], 2)})' if 'lift' in six else '') + '. Planning does not change: '
              + ', '.join(f'{f(pct(r["official"]["ntrl_mppi"]["success"], r["official"]["ntrl_mppi"]["n"]))} %' for n, r, k in ws) + ' of the runs solved.'
              + ((f' At weight {dict(w30=30, w100=100)[ws[-1][2]]} about {f(100 * (six["lbs"]["rows"]["1.0"]["ratio"][1] - six[ws[-1][2]]["rows"]["1.0"]["ratio"][1]) / (six["lbs"]["rows"]["1.0"]["ratio"][1] - six["lift"]["rows"]["1.0"]["ratio"][1]), 0)} % of the gap to the lift-only field is closed, '
                  f'and the steps next to obstacles go down with the rest (median {f(six[ws[-1][2]]["rows"]["1.0"]["groups"]["slow"]["ratio"], 2)} of the truth, LBS {f(six["lbs"]["rows"]["1.0"]["groups"]["slow"]["ratio"], 2)}, lift only {f(six["lift"]["rows"]["1.0"]["groups"]["slow"]["ratio"], 2)}): '
                  'the segment loss lowers the whole field toward the lift-only field; it does not make the times next to obstacles right.') if ws[-1][2] in ('w30', 'w100') and 'lift' in six and 'lbs' in six else '') + '</p>')
        vs = [(n, r, k) for n, r, k, s in have if k in ('lpath', 'lnet', 'bgoal')]
        if vs:
            A('<p><b>The leaf rule does not matter either:</b> ' + ', '.join(f'{n.split(",")[0]} {f(six[k]["rows"]["1.0"]["ratio"][1], 2)}' for n, r, k in vs if k != 'bgoal')
              + (f'. Letting the Bellman update also try the straight step to the goal gives {f(six["bgoal"]["rows"]["1.0"]["ratio"][1], 2)}' if 'bgoal' in six else '')
              + '. In the training drives the true time of the straight step across a leaf is 0.94 of the drive’s own time over it, so path leaves and min leaves are both close to the truth already; the leaves are not where the bias comes from.</p>')
        hf = {k: (r, six[k]['rows']['1.0']) for n, r, k, s in have if k in ('lbs', 'lift', 'lbsf05', 'liftf05')}
        if len(hf) == 4:
            so = lambda k: hf[k][0]['official']['ntrl_mppi']['success']  # noqa: E731
            s2r = [r for r in reps if r.get('scene') == 'real_world_6dof' and all(r['arm'].values()) and abs(r['freq'][0] - 1) < 1e-9]
            A(f'<p><b>Half the Fourier frequency lowers the overestimate and is the one change here that also improves planning.</b> LBS at frequency × 0.5 predicts {f(hf["lbsf05"][1]["ratio"][1], 2)} of the true time (frequency × 1: {f(hf["lbs"][1]["ratio"][1], 2)}), '
              f'{f(100 * hf["lbsf05"][1]["within"], 0)} % of the steps within 20 %, and it solves {so("lbsf05"):,} of the 1,800 runs, against {so("lbs"):,} at frequency × 1'
              + (' (' + ', '.join(f'{r["official"]["ntrl_mppi"]["success"]:,}' for r in s2r) + ' with another training seed)' if s2r else '')
              + f'. The lift alone at half the frequency: {f(hf["liftf05"][1]["ratio"][1], 2)} and {so("liftf05"):,} runs ({so("lift"):,} at frequency × 1). '
              'So at half the frequency LBS is ahead of the lift alone on this scene, by ' + f'{so("lbsf05") - so("liftf05")} runs. '
              + ('A second training seed of LBS at half the frequency: ' + ', '.join(f'{r["official"]["ntrl_mppi"]["success"]:,}' for r in reps_other if r['label'] == 'rw_lbs_f05_s2') + '. ' if any(r['label'] == 'rw_lbs_f05_s2' for r in reps_other)
                 else 'This is one training seed; a second is training. ')
              + ('The lift alone with the second seed at half the frequency: ' + ', '.join(f'{r["official"]["ntrl_mppi"]["success"]:,}' for r in reps_other if r['label'] == 'rw_lift_f05_s2')
                 + ', so LBS is ahead of the lift alone on both seeds. ' if any(r['label'] == 'rw_lift_f05_s2' for r in reps_other) else '')
              + ('The ntrl-demo control at half the frequency: ' + ', '.join(f'{r["official"]["ntrl_mppi"]["success"]:,}' for r in every if r['label'] == 'rw_ntrl_f05')
                 + ' runs, about what it solves at its own frequency; the lower frequency helps only with the lift. ' if any(r['label'] == 'rw_ntrl_f05' for r in every) else '')
              + (f'A quarter of the frequency: {next(r for n, r, k, s_ in have if k == "f025")["official"]["ntrl_mppi"]["success"]:,} runs and {f(six["f025"]["rows"]["1.0"]["ratio"][1], 2)} of the true time, so the values are closer still and planning is level with half the frequency. '
                 if any(k == 'f025' for n, r, k, s_ in have) else '')
              + 'On the cabinet half the frequency did not change the LBS result.</p>')
        BD = D.get('belldiag', {})
        if 'real_world_6dof' in BD and all(k in BD['real_world_6dof']['fields'] for k in ('lift', 'lbs')):
            b6 = BD['real_world_6dof']['fields']
            b12 = BD.get('real_world_12dof', {}).get('fields', {})
            m = lambda b, k, w='as trained': b[k][w]['mean_residual_radii']  # noqa: E731
            A(f'<p><b>Where the bias comes from.</b> The Bellman update compares T(a, goal) with the true cost of a short step plus T at the step’s end. Evaluated on the lift-only field, whose open-space times are right, '
              f'that target is on average {f(-m(b6, "lift"), 2)} step radii above the field’s value' + (f' ({f(-m(b12, "lift"), 2)} with two arms)' if 'lift' in b12 else '')
              + f'; adding the straight step to the goal barely moves it ({f(-m(b6, "lift", "goal step"), 2)}). The training pairs start next to obstacles, where the true step cost is high and every field here underestimates '
              'the time (see the close pairs by speed above). So the Bellman update is right to ask for more time there. My reading of what follows: the network does not put the extra time next to the obstacles; it raises the time everywhere until the average residual is zero '
              f'(LBS: {f(m(b6, "lbs"), 2)} radii), which is the overestimate in open space, while steps next to obstacles stay underestimated. '
              f'Mean T over the training pairs: lift only {f(b6["lift"]["as trained"]["T_mean"], 3)}, LBS {f(b6["lbs"]["as trained"]["T_mean"], 3)}'
              + (f'; two arms {f(b12["lift"]["as trained"]["T_mean"], 3)} and {f(b12["lbs"]["as trained"]["T_mean"], 3)}' if all(k in b12 for k in ('lift', 'lbs')) else '') + '. '
              'How far it raises it follows the Fourier frequency (chart above): on the cabinet, at half or a quarter of the frequency the LBS field is within 10 % of the truth; with two arms half the frequency halves the overestimate and does not remove it. '
              f'Runs <code>{BD["real_world_6dof"]["run"]}</code>' + (f', <code>{BD["real_world_12dof"]["run"]}</code>' if 'real_world_12dof' in BD else '') + '.</p>')
        pend = [q['label'] for q in D['pending'] if q['label'].startswith('rw_lbs_')]
        if pend:
            A('<p>Still training or queued: ' + ', '.join(vn[q][0] for q in pend if q in vn) + '.</p>')
        A(ch('leaf_ratio'))
        A(ch('leaf_solved'))
    if reps:
        key = lambda r: (r.get('scene', 'cabinet'), akey(r))  # noqa: E731
        firsts = {key(r): r for r in every if r['label'] in ('lbs', 'ntrl', 'rw_lbs', 'rw_ntrl')}
        rows = []
        # the runs at another frequency that have a second seed: label of the first seed -> (row name, label of the second seed)
        for l0, (nm_, l1) in {'lbsf025': ('LBS, frequency × 0.25', 'lbsf025_s2'), 'rw_lbs_f05': ('LBS, frequency × 0.5', 'rw_lbs_f05_s2'), 'rw_lift_f05': ('lift only, frequency × 0.5', 'rw_lift_f05_s2')}.items():
            r0 = next((r for r in every if r['label'] == l0), None)
            r1_ = next((r for r in reps_other if r['label'] == l1), None)
            if r0 and r1_:
                firsts[(r0.get('scene', 'cabinet'), nm_)] = [r0, r1_]
        for k, r0 in firsts.items():
            rs = r0 if isinstance(r0, list) else [r0] + [r for r in reps if key(r) == k]
            if len(rs) < 2:
                continue
            v = [pct(r['official']['ntrl_mppi']['success'], r['official']['ntrl_mppi']['n']) for r in rs]
            dv = [pct(*r['official']['ntrl_mppi']['detour']) for r in rs]
            rows.append(['cabinet' if k[0] == 'cabinet' else 'real-world cupboard, one arm', (k[1] if isinstance(r0, list) else 'LBS' if k[1] == 'LBS' else 'ntrl-demo (control)'), ', '.join(str(r['seed']) for r in rs), ', '.join(f(x) for x in v), f(sum(v) / len(v)),
                         ', '.join(f(x) for x in dv), ', '.join(f'<code>{r["id"][-8:]}</code>' for r in rs)])
        if rows:
            A('<h2 id="seeds">Other training seeds</h2>')
            A('<p>The first LBS run and its control repeated with other training seeds (same data, same queries, same planner seeds), and the runs at a lower Fourier frequency that have a second seed.</p>')
            A('<div class="sumtbl">' + table(['scene', 'method', 'training seeds', 'solved % of 1,800 runs', 'mean', 'detour runs solved %', 'field runs'], rows, left=2) + '</div>')
    A('<h2 id="setup">Setup</h2>')
    A('<ul><li><b>Scene and queries:</b> the table and cupboard of the cabinet task, the release UR5e with its gripper, the 600 frozen queries (six suites of 100). '
      'Final numbers come from the unchanged cabinet evaluator and release scorer (obstacle clearance at least 5 mm, self clearance at least 2 mm, joint limits).</li>'
      '<li><b>Coordinates:</b> each joint divided by its range, so every joint has range 1.</li>'
      '<li><b>Lift onto 7 dimensions:</b> the network’s input is the six joints and h, the travel-time height of the configuration computed from its clearance. '
      'At gain 1 the range of h is 0.12. The range of h and the frequency of its Fourier column are the same knob.</li>'
      '<li><b>Bellman:</b> the backup with the true step cost, the clearance evaluated exactly along every candidate step, on 512 of the 2,000 pairs of each minibatch.</li>'
      '<li><b>Segment:</b> the midpoint loss from epoch 2,000, 4,096 traversals per refresh between configurations at most 0.6 apart.</li>'
      '<li><b>Speed model:</b> the clearance is the obstacle distance, with self-contact as a thin-band wall. The release label (obstacle distance and half the self distance) is not used: '
      'with it the self term sets the speed of 86 % of the valid configurations.</li>'
      '<li><b>Planner:</b> ntrl-demo’s MPPI, 50 samples, horizon 5, 600 steps of 0.0034 (about 0.04 rad), joints clamped to their limits; the same for every run of this study.</li></ul>')
    A(details('Run ids of the earlier results', table(['method', 'registered run', 'planner settings'], [[esc(b['name']), f'<code>{b["id"]}</code>', esc(json.dumps(b['planner']))] for b in D['baselines']], left=1)))
    css = (LS / 'page_assets/style.css').read_text()
    js = (LS / 'page_assets/charts.js').read_text()
    page = (f'<title>{TITLE_TAG}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">\n'
            f'<style>\n{css}\n{EXTRA_CSS}</style>\n<div class="wrap">\n' + '\n'.join(H) + '\n</div>\n<div class="tip" id="tip" hidden></div>\n'
            f'<script id="data" type="application/json">{json.dumps(dict(charts=C))}</script>\n<script>\n{js}\n</script>\n')
    (OUT / 'lbs_arm.html').write_text(page)
    print(f'{OUT / "lbs_arm.html"}: {len(page) / 1024:.0f} KB, {len(C)} charts')


if __name__ == '__main__':
    main()
