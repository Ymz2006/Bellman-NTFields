"""Report page of the LBS benchmark: report/lbs.html from report/runs.json (report.py).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs/page.py

Chart code and style sheet are jeffrey/lift_scale/page_assets. Published as the artifact "2026-11-19 LBS Bellman Benchmarking".
"""
import importlib.util
import base64
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
LS = ROOT / 'jeffrey/lift_scale'
_spec = importlib.util.spec_from_file_location('lift_scale_pagelib', LS / 'pagelib.py')
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
f, hbar, line, table, details, chart, esc = L.f, L.hbar, L.line, L.table, L.details, L.chart, L.esc

TITLE_TAG = '2026-11-19 LBS Bellman Benchmarking'
ARMS = ['lbs', 'noL', 'noB', 'noS', 'ntrl']
NAME = dict(lbs='LBS', noL='no lift', noB='no Bellman', noS='no segment', ntrl='ntrl-demo', noLw='no lift with the Bellman weight × 0.27 (control)', lbsP='LBS with the plain, min leaves, tree segment')
BINS = ['0-4', '4-8', '8-16', '16-32', '32-64', '64-128']
QB = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128']
LONG = dict(lbs='LBS: lift + Bellman + segment', noL='no lift: Bellman + segment', noB='no Bellman: lift + segment', noS='no segment: lift + Bellman', ntrl='ntrl-demo')
SLOT = dict(lbs=4, noL=2, noB=3, noS=5, ntrl=1)
FAMILIES = [('mazes', 'Mazes', ['maze10', 'maze13', 'maze18']), ('gap', '10 × 10 gap mazes', ['gap1', 'gap2', 'gap4']), ('gibson', 'Gibson houses', ['Allensville', 'Lakeville'])]
G3 = ['Allensville3d', 'Mifflinburg3d', 'Noxapater3d', 'Markleeville3d', 'Lakeville3d']       # 3-D Gibson: LBS and ntrl-demo only
TYPES = [('Maze, 2-D', FAMILIES[0][2]), ('Gap maze, 2-D', FAMILIES[1][2]), ('Gibson, 2-D', FAMILIES[2][2]), ('Gibson, 3-D', G3)]
TITLE = dict(maze10='Maze 10 × 10', maze13='Maze 13 × 13', maze18='Maze 18 × 18', gap1='Gap maze ×1 (16 cells)', gap2='Gap maze ×2 (22 cells)', gap4='Gap maze ×4 (34 cells)',
             Allensville='Allensville (43 cells)', Lakeville='Lakeville (83 cells)', Klickitat='Klickitat (96 cells)', **{s: s[:-2] + ' in 3-D' for s in G3})
SHORT = dict(maze10='10 × 10', maze13='13 × 13', maze18='18 × 18', gap1='gap ×1', gap2='gap ×2', gap4='gap ×4', Allensville='Allensville', Lakeville='Lakeville', Klickitat='Klickitat', **{s: s[:-2] + ' 3-D' for s in G3})
EXTRA_CSS = '''
.status { display: grid; gap: 4px; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 16px; margin: 10px 0 4px; font-size: 14.5px; }
.status b { font-weight: 600; }
.envs { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 12px 0 6px; }
.envs figure { margin: 0; min-width: 0; }
.envs img { display: block; width: 100%; height: auto; aspect-ratio: 1; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
.envs figcaption { font-size: 13.5px; margin-top: 6px; }
.envs figcaption b { font-weight: 600; }
.envnote { font-size: 13.5px; opacity: .8; margin: 2px 0 12px; }
.one { max-width: 460px; }
.sumtbl { overflow-x: auto; margin: 10px 0 16px; }
.sumtbl table { min-width: 0; width: 100%; max-width: 640px; font-size: 14.5px; }
.sumtbl td, .sumtbl th { padding: 6px 10px; }
.arms { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); gap: 10px; margin: 10px 0 6px; }
.arms > div { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 12px; font-size: 13.5px; color: var(--ink-2); min-width: 0; }
.arms b { display: block; color: var(--ink); font-size: 14.5px; margin-bottom: 2px; }
.arms i { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
'''


class Data:
    def __init__(self):
        d = json.load(open(OUT / 'runs.json'))
        self.runs, self.pending, self.failed = d['runs'], d['pending'], d['failed']
        self.extra, self.controls = d.get('extra', []), d.get('controls', [])
        self.s3 = d.get('scenes3d', {})
        self.allens = json.load(open(OUT / 'allensville.json')) if (OUT / 'allensville.json').exists() else None

    def sel(self, scene, arm):
        return sorted([r for r in self.runs if r['scene'] == scene and r['arm'] == arm], key=lambda r: (r['seed'] != 87358, r['seed']))

    def one(self, scene, arm):
        rs = self.sel(scene, arm)
        return rs[0] if rs else None

    def var(self, scene):
        rs = [r for r in self.controls if r['scene'] == scene and r['arm'] == 'lbsP']
        return rs[0] if rs else None

    def pool(self, scene, arm, key='solved'):
        rs = self.sel(scene, arm)
        if not rs:
            return None
        if key == 'solved':
            return 100 * sum(r['success'][0] for r in rs) / sum(r['success'][1] for r in rs)
        v = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean(v)) if v else None


def img(name):
    f = OUT / f'env/{name}.png'
    return 'data:image/png;base64,' + base64.b64encode(f.read_bytes()).decode() if f.exists() else None


def env_row(scenes, three=False):
    meta = json.load(open(OUT / 'env.json')) if (OUT / 'env.json').exists() else {}
    figs = ''
    for s in scenes:
        src, m = img(s), meta.get(s)
        if src and m:
            figs += (f'<figure><img src="{src}" alt="{TITLE[s]}: obstacles seen from above with the true path of the longest test query">'
                     f'<figcaption><b>{TITLE[s]}</b><br>{m["queries"]} test queries; longest true path {m["longest_path_cells"]:.0f} cells</figcaption></figure>')
    if not figs:
        return ''
    if three:
        return (f'<div class="envs">{figs}</div><p class="envnote">Seen from above: how much of the 2 m slab above each spot is free air (white = all of it, dark = none; furniture shows as grey). '
                'Blue: the true path of the longest test query, from its start (dot) to its goal (ring), flattened onto the plan. The robot is a point.</p>')
    return (f'<div class="envs">{figs}</div><p class="envnote">Seen from above, the whole box: obstacles filled, free space left open. Blue: the true path of the longest test query, from its start (dot) to its goal (ring). '
            'The robot is a 0.5 × 0.2-cell rectangle.</p>')


def note(D, scene, arm):
    rs = D.sel(scene, arm)
    out = []
    if len(rs) > 1:
        out.append(f'{len(rs)} training seeds: ' + ' / '.join(f(r['solved']) for r in sorted(rs, key=lambda r: r['seed'])))
    if rs and rs[0]['source'] != 'jeffrey/lbs':
        out.append('run of ' + rs[0]['source'])
    return '; '.join(out)


def build_charts(D):
    C = {}
    leg = [(LONG[a], SLOT[a]) for a in ARMS]
    for key, fam, scenes in FAMILIES:
        for metric, title, sub, vn in (('solved', 'queries solved', 'ntrl-demo’s MPPI, planning seeds 0, 1 and 2, exact scoring; the network at epoch 5,000.', 'solved'),
                                       ('within10', 'travel times within 10 % of the truth', 'Random (free configuration, goal) pairs against the grid solver’s travel time.', 'within 10 %'),
                                       ('seen_through', 'detours seen through', 'Share of the pairs with a true detour whose learned time is close to the straight-line time (lower is better).', 'seen through')):
            groups = []
            for s in scenes:
                rows = [(NAME[a], D.pool(s, a, metric), SLOT[a], note(D, s, a)) for a in ARMS if D.pool(s, a, metric) is not None]
                if rows:
                    groups.append((TITLE[s], rows))
            if groups:
                C[f'{metric}_{key}'] = hbar(f'{fam}: {title}', sub, groups, legend=leg, value_name=vn)
    for _, _, scenes in FAMILIES:
        for s in scenes:
            ser_e, ser_m = [], []
            for a in ARMS:
                r = D.one(s, a)
                if not r or not r.get('course'):
                    continue
                ser_e.append((LONG[a], SLOT[a], [(c['epoch'], c['pct']) for c in r['course']], None))
                if all(c.get('minutes') is not None for c in r['course']):
                    ser_m.append((LONG[a], SLOT[a], [(c['minutes'], c['pct']) for c in r['course']], None))
            if ser_e:
                C[f'course_{s}'] = line(f'{TITLE[s]}: queries solved during training', 'Planning seed 0 on the test queries with the network of that epoch. The segment loss starts at epoch 2,000.', ser_e,
                                        xlabel='epoch', ylabel='solved %', xticks=[0, 1000, 2000, 3000, 4000, 5000], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, xmin=0, xmax=5000, digits=1, xname='epoch', height=260)
            if ser_m:
                xm = max(p[0] for _, _, pts, _ in ser_m for p in pts)
                step = 10 if xm <= 60 else 20 if xm <= 120 else 30
                C[f'clock_{s}'] = line(f'{TITLE[s]}: queries solved against training time', 'The same points against minutes of training on the GPU the run had (planning for these points not counted).', ser_m,
                                       xlabel='minutes of training', ylabel='solved %', xticks=list(range(0, int(xm) + step, step)), yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, xmin=0, digits=1, xname='minute', height=260)
    if D.allens:
        ser = []
        for r in D.allens['runs']:
            own = r['source'] == 'jeffrey/lbs'
            name = f"{NAME[r['arm']]}, seed {r['seed']}" + ('' if own else ', earlier study’s run')
            ser.append((name, SLOT[r['arm']] if own else 6, [(c['epoch'], 240 - c['fails']) for c in r['course']], None if r['seed'] == 87358 else '5 4'))
        C['allens_course'] = line('Allensville: queries solved during training, of 240', 'Planning seed 0, the network of that epoch. The runs of one seed share their first 2,000 epochs exactly; the segment loss starts there. '
                                  'Dashed: the other training seeds.', ser, xlabel='epoch', ylabel='queries solved of 240', xticks=[1000, 2000, 3000, 4000, 5000], yticks=[216, 222, 228, 234, 240], ymin=216, ymax=240,
                                  xmin=1000, xmax=5000, digits=0, xname='epoch', height=280)
    for s in ('Lakeville',):
        pair = [r for a in ('lbs', 'noS') for r in D.sel(s, a) if r['source'] == 'jeffrey/lbs']
        if len(pair) == 2:
            ser = [(NAME[r['arm']], SLOT[r['arm']], [(c['epoch'], c['success']) for c in r['course']], None) for r in pair]
            n = pair[0]['course'][0]['runs']
            lo = min(c['success'] for r in pair for c in r['course'])
            C[f'pair_{s}'] = line(f'{s}: queries solved during training, of {n}', 'Planning seed 0, the network of that epoch; the same training seed, code and GPU. The segment loss starts at epoch 2,000.', ser,
                                  xlabel='epoch', ylabel=f'queries solved of {n}', xticks=[1000, 2000, 3000, 4000, 5000], yticks=list(range((lo - 5) // 10 * 10, n + 1, 10)), ymin=(lo - 5) // 10 * 10, ymax=n, xmin=1000, xmax=5000, digits=0, xname='epoch', height=260)
    VL = [('lbs', 'LBS: safe, min leaves, tree + sub-paths', 4), ('lbsP', 'LBS: plain, min leaves, tree', 6), ('noS', 'no segment (reference)', 5)]
    for key, fam, scenes in FAMILIES:
        groups = []
        for s in scenes:
            if D.var(s):
                rows = [(n, r['solved'], sl, '') for a, n, sl in VL for r in ([D.var(s)] if a == 'lbsP' else [x for x in D.sel(s, a) if x['seed'] == 87358][:1])]
                groups.append((TITLE[s], rows))
        if groups:
            C[f'variant_{key}'] = hbar(f'{fam}: queries solved, by segment variant', 'Lift and Bellman the same in all three; training seed 87358; planning seeds 0, 1 and 2; the network at epoch 5,000.', groups,
                                       legend=[(n, sl) for _, n, sl in VL], value_name='solved')
        for s in scenes:
            if D.var(s):
                ser = [(n, sl, [(c['epoch'], c['pct']) for c in r['course']], None) for a, n, sl in VL for r in ([D.var(s)] if a == 'lbsP' else [x for x in D.sel(s, a) if x['seed'] == 87358][:1]) if r.get('course')]
                C[f'vcourse_{s}'] = line(f'{TITLE[s]}: queries solved during training, by segment variant', 'Planning seed 0, the network of that epoch. The three runs share their first 2,000 epochs when they ran on the same GPU.', ser,
                                         xlabel='epoch', ylabel='solved %', xticks=[0, 1000, 2000, 3000, 4000, 5000], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, xmin=0, xmax=5000, digits=1, xname='epoch', height=260)
    for s in FAMILIES[2][2]:
        ser = []
        rs = [(LONG[a], SLOT[a], D.one(s, a), None) for a in ARMS] + [(NAME[r['arm']], 6, r, '5 4') for r in D.controls if r['scene'] == s and r['arm'] == 'noLw']
        for name, slot, r, dash in rs:
            pts = [(i + 1, r['by_distance'].get(b)) for i, b in enumerate(BINS) if r and r.get('by_distance', {}).get(b) is not None]
            if pts:
                ser.append((name, slot, pts, dash))
        if ser:
            n = max(p[0] for _, _, pts, _ in ser for p in pts)
            C[f'far_{s}'] = line(f'{s}: learned travel time over the true one, by true distance', 'Median over random (free configuration, goal) pairs whose true travel distance falls in the bin; 1 is exact, below 1 is too short.', ser,
                                 xlabel='true distance, cells', ylabel='median learned / true', xticks=list(range(1, n + 1)), xticklabels={str(i + 1): b.replace('-', '–') for i, b in enumerate(BINS)},
                                 yticks=[0, 0.25, 0.5, 0.75, 1.0], ymin=0, ymax=1.15, xmin=1, xmax=n, digits=2, xname='bin', height=260)
    groups = []
    for s in G3:
        rows = [(NAME[a], D.pool(s, a), SLOT[a], '') for a in ('ntrl', 'lbs') if D.sel(s, a)]
        if s in D.s3 and rows:
            rows.append(('straight line (control)', D.s3[s]['straight'], 6, 'the same queries, no planner'))
        if rows:
            groups.append((TITLE[s], rows))
        ser = [(LONG[a], SLOT[a], [(c['epoch'], c['pct']) for c in D.one(s, a)['course']], None) for a in ('ntrl', 'lbs') if D.one(s, a) and D.one(s, a).get('course')]
        if ser:
            C[f'course_{s}'] = line(f'{TITLE[s]}: queries solved during training', 'Planning seed 0, the network of that epoch.', ser, xlabel='epoch', ylabel='solved %',
                                    xticks=[0, 1000, 2000, 3000, 4000, 5000], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, xmin=0, xmax=5000, digits=1, xname='epoch', height=260)
    if groups:
        C['solved_g3d'] = hbar('Gibson houses in 3-D: queries solved', 'Point robot in R³; ntrl-demo’s MPPI, planning seeds 0, 1 and 2, exact scoring; the network at epoch 5,000; training seed 87358.', groups,
                               legend=[(LONG['ntrl'], SLOT['ntrl']), (LONG['lbs'], SLOT['lbs']), ('straight line (control)', 6)], value_name='solved')
    return C


def body(D, C):
    H = []
    A = H.append
    ch = lambda k: chart(k) if k in C else ''  # noqa: E731

    def grid(*keys):
        inner = ''.join(ch(k) for k in keys)
        return f'<div class="grid2">{inner}</div>' if inner else ''

    def v(scene, arm, key='solved', d=1):
        return f(D.pool(scene, arm, key), d)
    n_done = sum(1 for _, _, sc in FAMILIES for s in sc for a in ARMS if D.sel(s, a))
    n_all = 5 * sum(len(sc) for _, _, sc in FAMILIES)

    def series(scenes, arm):
        return ' / '.join(v(s, arm) for s in scenes)

    def fam_done(scenes, arm='lbs'):
        return all(D.sel(s, arm) for s in scenes)
    # ---------------------------------------------------------------- header
    lede = []
    for key, fam, scenes in FAMILIES:
        have = [s for s in scenes if D.sel(s, 'lbs')]
        if not have:
            continue
        txt = f'<b>{fam}</b> ({", ".join(SHORT[s] for s in have)}): LBS <b>{series(have, "lbs")} %</b> of the queries solved, ntrl-demo {series(have, "ntrl")} %'
        for a in ('noL', 'noB', 'noS'):
            if any(D.sel(s, a) for s in have):
                txt += f', {NAME[a]} {series(have, a)} %'
        lede.append(txt + '.')
    A('<header class="top">'
      '<div class="eyebrow">jeffrey/lbs · LBS = lift + Bellman + segment · rectangle robot in SE(2), ntrl-demo’s MPPI, exact scoring · every number from registered runs</div>'
      '<h1>LBS Bellman benchmarking: lift + Bellman + segment, with each part removed in turn</h1>'
      '<p class="lede">LBS adds three things to ntrl-demo: the travel-time <b>l</b>ift of the input, a <b>B</b>ellman backup over short collision-free steps priced by the true speed, and the '
      'travel-time midpoint (<b>s</b>egment) loss on paths the planner drives. Every scene is trained five ways: ntrl-demo, LBS, and LBS with one part removed. One training seed (87358) per arm so far. '
      + ' '.join(lede) + '</p></header>')
    run_now = [p for p in D.pending if p.get('epoch') and p['arm'] != 'lbsP']
    A('<div class="status">'
      f'<div><b>{n_done} of {n_all}</b> scene × arm results are in' + (f'; {len(D.failed)} run(s) failed' if D.failed else '') + '.</div>'
      + (f'<div>Training now: ' + ', '.join(f'{NAME[p["arm"]]} on {SHORT[p["scene"]]}' + (f', seed {p["seed"]}' if p['seed'] != 87358 else '') + f' (epoch {p["epoch"]:,} of 5,000)' for p in run_now) + '.</div>' if run_now else '')
      + (f'<div>Queued: ' + ', '.join(f'{NAME[p["arm"]]} · {SHORT[p["scene"]]}' + (f', seed {p["seed"]}' if p['seed'] != 87358 else '') for p in D.pending if not p.get('epoch') and p['arm'] != 'lbsP') + '.</div>' if any(not p.get('epoch') and p['arm'] != 'lbsP' for p in D.pending) else '')
      + '</div>')

    # ---------------------------------------------------------------- summary by test type (asked 2026-11-20)
    def each(pairs):
        return ', '.join(f'{SHORT[s]} {v:+.1f}' for s, v in pairs).replace('-', '−')
    A('<h2 id="summary">Summary by test type: baseline and LBS</h2>')
    A('<p>Queries solved, planning seeds 0, 1 and 2 pooled, the network at epoch 5,000, training seed 87358. Baseline = the original ntrl-demo. An empty cell is a run that has not finished. '
      'Klickitat was taken out of the benchmark on 2026-11-20.</p>')
    rows = []
    for tname, scenes in TYPES:
        vals = {a: [D.one(s, a)['solved'] if D.one(s, a) else None for s in scenes] for a in ('ntrl', 'lbs')}       # training seed 87358
        for i, s in enumerate(scenes):
            cells = [('' if vals[a][i] is None else f(vals[a][i]) + ' %') for a in ('ntrl', 'lbs')]
            rows.append([f'<b>{tname}</b>' if i == 0 else '', TITLE[s], cells[0], f'<b>{cells[1]}</b>' if cells[1] else ''])
        if all(v is not None for a in vals for v in vals[a]):
            rows.append(['', '<i>average</i>', f'<i>{f(float(np.mean(vals["ntrl"])))} %</i>', f'<i><b>{f(float(np.mean(vals["lbs"])))} %</b></i>'])
    A('<div class="sumtbl">' + table(['test type', 'scene', 'baseline (ntrl-demo)', 'LBS'], rows, left=2) + '</div>')

    # ---------------------------------------------------------------- Gibson in 3-D (asked 2026-11-20)
    g3_jobs = [(s, a) for s in G3 for a in ('ntrl', 'lbs')]
    g3_done = [(s, a) for s, a in g3_jobs if D.sel(s, a)]
    g3_pend = [p_ for p_ in D.pending if p_['scene'] in G3]
    A('<h2 id="gibson3d">Gibson houses in 3-D: a point robot in R³</h2>')
    A('<p>Asked on 2026-11-20: the first 3-D test. Five Gibson houses (Allensville and Lakeville from the 2-D set, plus Mifflinburg, Noxapater and Markleeville), LBS against ntrl-demo only. '
      'The robot is a point that flies: its configuration is (x, y, z). Each house is the scan’s mesh at 2.5 cm voxels inside a slab of the main floor, 0.15 to 2.15 m above the floor '
      '(one cell is 0.25 m as in the 2-D houses, so the slab is 8 cells high). Every voxel that holds a piece of the mesh is an obstacle, and so is the outside of the house and everything below and above the slab; '
      'free space is the connected air of the rooms, so a path can go over or under furniture and has to go through the doors.</p>'
      '<p><b>The lift is onto four dimensions:</b> the network’s input is (x, y, z, h), with h the travel-time height of the point computed from its clearance, exactly the lift of the 2-D tests, '
      'whose input is (x, y, heading, h). The rule in both is input dimensions + 1. Bellman and segment are unchanged. Planner, scorer, training length and seed are those of the 2-D houses.</p>')
    A('<div class="status">' + f'<div><b>{len(g3_done)} of {len(g3_jobs)}</b> runs are in; {len(D.s3)} of {len(G3)} houses are prepared (voxels, training pairs, ground truth, test queries).</div>'
      + (f'<div>Training now: ' + ', '.join(f'{NAME[p_["arm"]]} on {SHORT[p_["scene"]]} (epoch {p_["epoch"]:,} of 5,000)' for p_ in g3_pend if p_.get('epoch')) + '.</div>' if any(p_.get('epoch') for p_ in g3_pend) else '')
      + (f'<div>Queued: ' + ', '.join(f'{NAME[p_["arm"]]} · {SHORT[p_["scene"]]}' for p_ in g3_pend if not p_.get('epoch')) + '.</div>' if any(not p_.get('epoch') for p_ in g3_pend) else '') + '</div>')
    A(env_row([s for s in G3 if s in D.s3], three=True))
    A(ch('solved_g3d'))
    if g3_done:
        both = [s for s in G3 if D.sel(s, 'lbs') and D.sel(s, 'ntrl')]
        if both:
            A('<ul><li><b>LBS minus ntrl-demo, points of queries solved:</b> ' + each([(s, D.pool(s, 'lbs') - D.pool(s, 'ntrl')) for s in both]) + '.</li>'
              + '<li><b>How the failures split</b> (through a wall / stalled, % of runs): '
              + '; '.join(f'{SHORT[s]}: ntrl-demo {f(D.one(s, "ntrl")["wall"])} / {f(D.one(s, "ntrl")["stall"])}, LBS {f(D.one(s, "lbs")["wall"])} / {f(D.one(s, "lbs")["stall"])}' for s in both) + '.</li>'
              + '<li><b>Controls on the same queries:</b> a straight line solves ' + ', '.join(f'{SHORT[s]} {f(D.s3[s]["straight"])} %' for s in both if s in D.s3)
              + '; the same planner driven by the true travel time solves ' + ', '.join(f'{SHORT[s]} {f(float(np.mean(D.s3[s]["oracle"])))} %' for s in both if s in D.s3 and D.s3[s]['oracle'])
              + '.</li>'
              + ((f'<li><b>Reading:</b> LBS is ahead of ntrl-demo on all {len(both)} houses, by {min(D.pool(s, "lbs") - D.pool(s, "ntrl") for s in both):.1f} to {max(D.pool(s, "lbs") - D.pool(s, "ntrl") for s in both):.1f} points '
                  f'(average {f(float(np.mean([D.pool(s, "ntrl") for s in both])))} % → {f(float(np.mean([D.pool(s, "lbs") for s in both])))} %). ntrl-demo fails by arriving through walls; LBS almost never does and fails by stalling, '
                  f'mostly on the long queries: of the 32–64 cell queries it solves ' + ', '.join(f'{SHORT[s]} {f(D.one(s, "lbs")["by_bin"].get("32-64"), 0)} %' for s in both)
                  + ' (ntrl-demo ' + ', '.join(f'{f(D.one(s, "ntrl")["by_bin"].get("32-64"), 0)} %' for s in both) + '). '
                  'The planner on the true travel time solves 93 to 99 %, so the queries themselves are solvable; why LBS stalls more on Allensville and Lakeville than on the other three is not tested.</li>') if len(both) == len(G3) else '')
              + '<li><b>One training seed per arm.</b></li></ul>')
        cs = [f'course_{s}' for s in G3 if f'course_{s}' in C]
        for i in range(0, len(cs), 2):
            A(grid(*cs[i:i + 2]))
        rows = []
        for s in G3:
            for a in ('ntrl', 'lbs'):
                for r in D.sel(s, a):
                    rows.append([TITLE[s], LONG[a], f'{f(r["solved"])} ({r["success"][0]}/{r["success"][1]})', f(r['wall']), f(r['stall']), f(r['within10']), f(r['ratio'], 3), f(r['seen_through']), f(r['train_min'], 0), f'<code>{r["id"]}</code>'])
        nop = [(s, a, r['no_path_share']) for s in G3 for a in ('ntrl', 'lbs') for r in D.sel(s, a) if r.get('no_path_share', 0) > 0.05]
        A(details('3-D houses: the numbers', table(['scene', 'arm', 'solved %', 'through a wall %', 'stalled %', 'T within 10 %', 'median T / T*', 'detours seen through %', 'training min', 'registered run'], rows, left=2)
                  + ('<p>The travel-time columns compare random (free point, goal) pairs with the grid solver. Pairs that the solver’s grid does not connect have no true travel time and are left out: '
                     + ', '.join(f'{f(x)} % of the pairs for {NAME[a]} on {SHORT[s]}' for s, a, x in nop)
                     + '. Where more than 1 % of the pairs are left out, the median and the detour column cannot be recomputed from the stored numbers and are empty. The solved rates are not affected: every test query has a certified true path.</p>' if nop else '')))
        rows = []
        for s in G3:
            for a in ('ntrl', 'lbs'):
                r = D.one(s, a)
                if r and r.get('by_bin'):
                    rows.append([TITLE[s], NAME[a]] + [f(r['by_bin'].get(b), 0) for b in ('0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128')])
        A(details('3-D houses: queries solved by true path length (cells)', table(['scene', 'arm', '0.5–2', '2–4', '4–8', '8–16', '16–32', '32–64', '64–128'], rows, left=2)))
    if D.s3:
        rows = [[TITLE[s], f'{m["cells"]:.0f} × {m["cells"]:.0f} × {m["z_cells"]:.0f}', f'{m["side_m"]:.2f}', f(m['free_share']), m['queries'], f(m['longest_path_cells'], 0), f(m['straight']),
                 ' / '.join(f(o) for o in m['oracle']) or '–', ' × '.join(map(str, m['truth_grid'])), f'<code>{m["prepare"]}</code>'] for s, m in ((s, D.s3[s]) for s in G3 if s in D.s3)]
        A(details('3-D houses: the scenes', table(['scene', 'cells (x × y × z)', 'box side, m', 'free share of the slab %', 'test queries', 'longest true path, cells', 'straight line solves %',
                                                    'planner on the true field solves % (seeds 0 / 1 / 2)', 'truth grid nodes', 'prepare run'], rows, left=1)
                  + '<p>Ground truth: shortest travel time on a grid with 5 nodes per cell in all three directions, every grid edge that touches an obstacle removed '
                  '(the scans’ walls are as thin as one voxel, 0.1 cells). Queries: 40 per true-distance bin, each certified by a collision-free true path. <code>jeffrey/lbs/scenes3.py</code>, <code>prepare3.py</code>.</p>'
                  + ('<p><b>A cut-off part on Lakeville.</b> The scene’s free space is one connected piece at the 2.5 cm voxels, but on Lakeville the truth grid does not connect about 4 % of it to the rest '
                     '(an opening too narrow for the grid’s edges, which are 5 cm long and are removed when they touch a wall). Three of the 48 truth goals and 4 of the 280 test queries lie inside that part; '
                     'no test query crosses into it. The other four houses have no such part larger than 0.2 % of the pairs.</p>' if 'Lakeville3d' in D.s3 else '')))

    # ---------------------------------------------------------------- arms
    # ---------------------------------------------------------------- takeaways
    allsc = [s for _, _, sc in FAMILIES for s in sc]
    done = [s for s in allsc if D.sel(s, 'lbs')]

    def own(s, a):
        return bool(D.sel(s, a)) and D.sel(s, a)[0]['source'] == 'jeffrey/lbs'

    def worth(a, matched=None):
        out = []
        for s in done:
            if D.sel(s, a) and (matched is None or own(s, a) == matched):
                out.append((s, D.pool(s, 'lbs') - D.pool(s, a)))
        return out

    def rng(vals, d=1):
        return f'{min(vals):.{d}f}' if abs(max(vals) - min(vals)) < 0.05 else f'{min(vals):.{d}f} to {max(vals):.{d}f}'

    def each(pairs):
        return ', '.join(f'{SHORT[s]} {v:+.1f}' for s, v in pairs).replace('-', '−')

    def house_lift(wh):
        pos, neg = [x for x in wh if x[1] >= 3], [x for x in wh if x[1] < 3]
        if pos and neg:
            return (f' <b>On the houses it is needed on the larger one{"s" if len(pos) > 1 else ""}</b> ({each(pos)}) and not on the smallest ({each(neg)}). Without lift the larger house{"s fail" if len(pos) > 1 else " fails"} by stalling '
                    f'({rng([D.pool(s, "noL", "stall") for s, _ in pos])} % of the runs), not through walls, and end{"" if len(pos) > 1 else "s"} below ntrl-demo: the travel times stop growing beyond about 8 to 16 cells '
                    '(<a href="#lift-houses">the lift section</a>).')
        if neg:
            return f' On the houses so far it is not needed ({each(neg)}); see <a href="#allensville">the house section</a>.'
        return f' On the houses: {each(pos)}.' if pos else ''
    if len(done) >= 3:
        A('<h2>What the results show</h2><ul>')
        lv, nv = [D.pool(s, 'lbs') for s in done], [D.pool(s, 'ntrl') for s in done if D.sel(s, 'ntrl')]
        A(f'<li><b>LBS solves {rng(lv)} % of the queries on the {len(done)} scenes finished so far; ntrl-demo {rng(nv)} %.</b> LBS is ahead of ntrl-demo on every one of them, by {rng([d for _, d in worth("ntrl")])} points.</li>')
        wl, wb = worth('noL'), worth('noB')
        if wl:
            wm_, wh_ = [x for x in wl if x[0] not in FAMILIES[2][2]], [x for x in wl if x[0] in FAMILIES[2][2]]
            if wm_:
                A(f'<li><b>Lift</b> is worth {rng([d for _, d in wm_])} points on the mazes and gap mazes ({each(wm_)}). Without it the failures are plans through a wall: '
                  f'{rng([D.pool(s, "noL", "wall") for s, _ in wm_])} % of the runs, against {rng([D.pool(s, "lbs", "wall") for s, _ in wm_])} % for LBS.'
                  + house_lift(wh_) + '</li>')
        if wb:
            A(f'<li><b>Bellman</b> is worth {rng([d for _, d in wb])} points ({each(wb)}). Without it the failures are stalls: {rng([D.pool(s, "noB", "stall") for s, _ in wb])} % of the runs, '
              f'against {rng([D.pool(s, "lbs", "stall") for s, _ in wb])} % for LBS. Within the regular mazes and within the gap mazes, both parts are worth more as the scene grows.</li>')
        wm, wr = worth('noS', True), worth('noS', False)
        if wm:
            hs = [(s_, v) for s_, v in wm if s_ in FAMILIES[2][2]]
            A(f'<li><b>Segment</b> is the smallest of the three and not always a gain: {each(wm)} points (LBS minus no segment, both trained here). It helps on the regular mazes and the widest gap maze and '
              'has nothing left to add where the arm without it already solves everything. '
              + ((f'<b>On the houses the sign is mixed</b> ({each(hs)}; matched pairs, one seed each except Allensville): see <a href="#allensville">the house section</a>.'
                  if min(v for _, v in hs) < 0 < max(v for _, v in hs) else f'<b>On the houses</b>: {each(hs)}; see <a href="#allensville">the house section</a>.') if hs else '')
              + (f' Against reused runs of other code, not a matched control: {each(wr)}.' if wr else '') + '</li>')
        A('<li><b>One training seed per arm</b> (two on Allensville for LBS and no segment). Differences of a few points between arms are within what a second seed can move.</li></ul>')
        rows = [[TITLE[s], f(D.pool(s, 'lbs'))] + [(f'{D.pool(s, "lbs") - D.pool(s, a):+.1f}'.replace('-', '−') + ('' if own(s, a) else ' (reused run)') if D.sel(s, a) else 'pending') for a in ('noL', 'noB', 'noS', 'ntrl')] for s in done]
        A(details('What each part is worth on every scene: LBS minus the arm without it, points of queries solved',
                  table(['scene', 'LBS solved %', 'lift (vs no lift)', 'Bellman (vs no Bellman)', 'segment (vs no segment)', 'all three (vs ntrl-demo)'], rows)))
    # ---------------------------------------------------------------- segment variants
    vp = [p_ for p_ in D.pending if p_['arm'] == 'lbsP']
    vd = [s for s in allsc if D.var(s)]
    if vd or vp:
        A('<h2 id="variants">Segment variants inside LBS</h2>')
        A('<p>Asked on 2026-11-20: put two variants of the <a href="https://claude.ai/artifact/L7QtsFtB8Kk6nGv3mJ8d4U">travel-time midpoint loss</a> in as the segment part of LBS, without ablations. '
          '<b>Safe, min leaves, tree + sub-paths</b> is that page’s main setting and is what every LBS run here already uses, so the LBS results are that test. '
          '<b>Plain, min leaves, tree</b> changes two things: the training drives are the test planner’s own (nothing keeps a rollout 0.02 cells off the walls; a drive that touches a wall is cut there), '
          'and the loss uses only the pieces of the midpoint tree, without the random sub-paths between leaf ends. Lift, Bellman, training seed and GPU are those of the scene’s LBS run, so the pair is the same run up to epoch 2,000. '
          'The run without the segment loss is shown as the reference.</p>')
        A('<div class="status">' + f'<div><b>{len(vd)} of {len(allsc)}</b> plain-variant runs are in.</div>'
          + (f'<div>Training now: ' + ', '.join(f'{SHORT[p_["scene"]]} (epoch {p_["epoch"]:,} of 5,000)' for p_ in vp if p_.get('epoch')) + '.</div>' if any(p_.get('epoch') for p_ in vp) else '')
          + (f'<div>Queued: ' + ', '.join(SHORT[p_['scene']] for p_ in vp if not p_.get('epoch')) + '.</div>' if any(not p_.get('epoch') for p_ in vp) else '') + '</div>')
        if vd:
            for key, _, _ in FAMILIES:
                A(ch(f'variant_{key}'))
            safe = {s: [x for x in D.sel(s, 'lbs') if x['seed'] == 87358][0] for s in vd}
            late = lambda r: sum(c['pct'] for c in r['course'] if c['epoch'] >= 3000) / max(1, sum(1 for c in r['course'] if c['epoch'] >= 3000))
            ref = {s: ([x for x in D.sel(s, 'noS') if x['seed'] == 87358] or [None])[0] for s in vd}
            A('<ul><li><b>Plain minus safe, points of queries solved:</b> ' + each([(s, D.var(s)['solved'] - safe[s]['solved']) for s in vd]) + '.</li>'
              + '<li><b>Plain minus no segment:</b> ' + each([(s, D.var(s)['solved'] - ref[s]['solved']) for s in vd if ref[s]]) + '.</li>'
              + '<li><b>The training drives.</b> Share of the 8,192 drives per refresh that reach their goal, plain / safe: '
              + ', '.join(f'{SHORT[s]} {f(D.var(s)["drives"]["reached"], 0)} / {f(safe[s]["drives"]["reached"], 0)} %' for s in vd if D.var(s).get('drives') and safe[s].get('drives'))
              + '; cut at a wall: ' + ', '.join(f'{SHORT[s]} {f(D.var(s)["drives"]["collided"], 1)} / {f(safe[s]["drives"]["collided"], 1)} %' for s in vd if D.var(s).get('drives') and safe[s].get('drives'))
              + '. Pieces in the loss per refresh (thousands): '
              + ', '.join(f'{SHORT[s]} {D.var(s)["drives"]["pieces"] / 1000:.0f} / {safe[s]["drives"]["pieces"] / 1000:.0f}' for s in vd if D.var(s).get('drives') and safe[s].get('drives')) + '.</li>'
              + '<li><b>Movement between saved epochs</b> (planning seed 0, lowest to highest after epoch 2,000), plain / safe: '
              + ', '.join(f'{SHORT[s]} {min(c["pct"] for c in D.var(s)["course"] if c["epoch"] > 2000):.0f}–{max(c["pct"] for c in D.var(s)["course"] if c["epoch"] > 2000):.0f} / '
                          f'{min(c["pct"] for c in safe[s]["course"] if c["epoch"] > 2000):.0f}–{max(c["pct"] for c in safe[s]["course"] if c["epoch"] > 2000):.0f} %' for s in vd) + '. The bars above are the last epoch only.</li>'
              + '<li><b>Average over the saved epochs from 3,000 on</b> (planning seed 0; evens out which epoch the run happened to stop on), plain / safe / no segment: '
              + ', '.join(f'{SHORT[s]} ' + ' / '.join(f(late(r), 1) if r else '–' for r in (D.var(s), safe[s], ref[s])) + ' %' for s in vd) + '.</li>'
              + '<li><b>One training seed per variant.</b> On the houses a matched pair moved by about ±6 points from one saved epoch to the next; read differences of that size as noise.</li></ul>')
            cs = [f'vcourse_{s}' for s in vd if f'vcourse_{s}' in C]
            for i in range(0, len(cs), 2):
                A(grid(*cs[i:i + 2]))
            rows = []
            for s in vd:
                for name, r in (('safe, min leaves, tree + sub-paths (LBS)', safe[s]), ('plain, min leaves, tree', D.var(s)), ('no segment', ref[s])):
                    if r:
                        rows.append([TITLE[s], name, f'{f(r["solved"])} ({r["success"][0]}/{r["success"][1]})', f(late(r), 1), f(r['wall']), f(r['stall']), f(r['within10']), f(r['ratio'], 3), f(r['seen_through']), f(r['train_min'], 0), f'<code>{r["id"]}</code>'])
            A(details('Segment variants: the numbers', table(['scene', 'segment part', 'solved %', 'mean of saved epochs ≥ 3,000 (seed 0) %', 'through a wall %', 'stalled %', 'T within 10 %', 'median T / T*', 'detours seen through %', 'training min', 'registered run'], rows, left=2)))

    A('<h2>The method and the five arms</h2>')
    A('<div class="arms">'
      + ''.join(f'<div><b><i style="background: var(--s{SLOT[a]})"></i>{esc(NAME[a])}</b>{esc(t)}</div>' for a, t in (
          ('lbs', 'Lift + Bellman + segment: everything.'), ('noL', 'Bellman + segment on ntrl-demo’s own input.'), ('noB', 'Lift + segment; no backup.'),
          ('noS', 'Lift + Bellman; no midpoint loss.'), ('ntrl', 'The original baseline: none of the three.')))
      + '</div>')
    A('<p><b>L, lift.</b> The network gets one more input column, the travel-time height of the configuration, h(q) = G(d(q)) / E with G(d) the integral of √(1 / S² − 1) from the clearance d to the speed margin '
      '(jeffrey/lift_scale; no push, no octaves, gain 1).</p>'
      '<p><b>B, Bellman.</b> For each end q of a training pair (q, g): y = min over 16 straight steps q → q′ that stay out of obstacles of [τ(q, q′) + T(q′, g)], loss (T(q, g) − y)², from epoch 1. '
      'As asked, the backup uses the ground truth: τ is the step’s travel time at the true speed (its lifted length) and the steps are checked against the true geometry. '
      'Steps of 0.33–0.66 cells; on scenes larger than 24 cells (gap ×4 and the houses) lengths up to 5.3 cells. ntrl-demo’s own TD term skips steps that cross an obstacle (jeffrey/pair_lift’s recipe).</p>'
      '<p><b>S, segment.</b> The travel-time midpoint loss of the <a href="https://claude.ai/artifact/L7QtsFtB8Kk6nGv3mJ8d4U">midpoint page</a>, its main setting: from epoch 2,000, every 100 epochs the current field '
      'drives 8,192 random pairs with ntrl-demo’s MPPI; each drive is split recursively at travel-time midpoints down to one-cell leaves, and the network’s time over a piece is pulled to the sum over its leaves '
      '(|log T − log target|, weight ramped over 500 epochs).</p>'
      '<p>All arms: ntrl-demo’s network, loss, data (500,000 pairs) and 5,000 epochs; the rectangle robot (0.5 × 0.2 cells) in SE(2); the scene’s certified test queries, ntrl-demo’s MPPI with planning seeds 0, 1 and 2, '
      'a run solved when it reaches the goal without touching an obstacle.</p>')

    # ---------------------------------------------------------------- families
    for key, fam, scenes in FAMILIES:
        if not any(D.sel(s, a) for s in scenes for a in ARMS) and not any(f'course_{s}' in C for s in scenes):
            continue
        A(f'<h2>{fam}</h2>')
        if key == 'mazes':
            A('<p>maze_scale’s regular mazes with the 10 × 10 and 13 × 13 layouts of the earlier studies and a new 18 × 18 one; 200–240 test queries, 40 per true-distance bin.</p>')
        elif key == 'gap':
            A('<p>The 10 × 10 maze cut into four 5 × 5 quadrants pushed apart by an open band 6, 12 and 24 cells wide (×1, ×2, ×4): 16, 22 and 34 cells a side; 300 test queries.</p>')
        else:
            reused = [f'{NAME[a]} on {SHORT[s]}' for a in ARMS for s in scenes if D.sel(s, a) and D.sel(s, a)[0]['source'] != 'jeffrey/lbs']
            A('<p>Floor plans of two Gibson houses (the main floor’s slice; the same rectangle robot), 240–280 test queries. Klickitat (96 cells) was taken out of the benchmark on 2026-11-20; its runs stay registered. '
              + ('Reused from the lift-scale, lift-midpoint and pair-lift studies (the same scenes, queries, planner and recipe, but other code and random streams): ' + ', '.join(reused) + '. ' if reused else '')
              + 'A reused run is not a matched control: on Allensville the reused no-segment run scored 100.0 % and this benchmark’s own run of the same recipe and seed 94.2 % '
              '(<a href="#allensville">below</a>). Everything else is trained here.</p>')
        A(env_row(scenes))
        A(ch(f'solved_{key}'))
        rows = []
        for s in scenes:
            if D.sel(s, 'lbs'):
                base = D.pool(s, 'lbs')
                rows.append([TITLE[s], f(base)] + [(f'{base - D.pool(s, a):+.1f}' if D.pool(s, a) is not None else '–') for a in ('noL', 'noB', 'noS', 'ntrl')])
        if rows:
            A('<p>What each part is worth: LBS minus the arm without it, in points of queries solved (positive: the part helps).</p>')
            A('<div class="tbl" style="overflow-x:auto">' + table(['scene', 'LBS solved %', 'lift (vs no lift)', 'Bellman (vs no Bellman)', 'segment (vs no segment)', 'all three (vs ntrl-demo)'], rows) + '</div>')
        cs = [f'course_{s}' for s in scenes if f'course_{s}' in C]
        if cs:
            A('<h3>Success as training goes on</h3>')
            for i in range(0, len(cs), 2):
                A(grid(*cs[i:i + 2]))
            ck = [f'clock_{s}' for s in scenes if f'clock_{s}' in C]
            if ck:
                A('<p>The same curves against training time. The Bellman term makes an epoch 3–4 times longer than ntrl-demo’s; the segment loss adds a planning pass every 100 epochs after epoch 2,000.</p>')
                for i in range(0, len(ck), 2):
                    A(grid(*ck[i:i + 2]))
        A(grid(f'within10_{key}', f'seen_through_{key}'))

    if D.allens:
        AL = D.allens
        R = {(r['arm'], r['seed'], r['source'] == 'jeffrey/lbs'): r for r in AL['runs']}
        l0, n0, old = R.get(('lbs', 87358, True)), R.get(('noS', 87358, True)), R.get(('noS', 87358, False))
        if l0 and n0 and old:
            A('<h2 id="allensville">The houses: does the segment loss hurt?</h2><h3>Allensville</h3>')
            A(f'<p>The first LBS run on Allensville solved {f(l0["solved"])} %, below the {f(old["solved"])} % of the no-segment run reused from the pair-lift study. Asked to make sure the segment loss is really harmful there, '
              f'I re-ran the no-segment arm with this benchmark’s code, the same training seed and the same GPU as the LBS run: <b>{f(n0["solved"])} %</b>. The two runs are identical through epoch 2,000, where the segment loss starts.</p>')
            A(ch('allens_course'))
            if img('Allensville_room'):
                A(f'<div class="envs one"><figure><img src="{img("Allensville_room")}" alt="The top-left part of Allensville with the starts of the failed queries and the spots where their plans stop">'
                  '<figcaption><b>Where the same-seed pair fails</b> (top-left part of the house). Circles: starts of the failed queries; crosses: where their plans stop. Pink: no segment; orange: LBS '
                  '(a pink circle covers the orange one where both fail). Blue frame: the room. The plans stop in open floor just outside it, not at a wall.</figcaption></figure></div>')
            ep = lambda r, e: next((c for c in r['course'] if c['epoch'] == e), None)  # noqa: E731
            l1, n1 = R.get(('lbs', 1, True)), R.get(('noS', 1, True))
            if l1 and n1:
                swing = max(abs(c2['fails'] - c1['fails']) for r in (l0, n0, l1, n1) for c1, c2 in zip(r['course'], r['course'][1:]))
                d0, d1 = l0['fails_per_planning_seed'][0] - n0['fails_per_planning_seed'][0], l1['fails_per_planning_seed'][0] - n1['fails_per_planning_seed'][0]
                second = (f'<li><b>Second training seed: no segment {f(n1["solved"])} %, LBS {f(l1["solved"])} %</b> ({n1["fails_per_planning_seed"][0]} and {l1["fails_per_planning_seed"][0]} failed of 240; '
                          f'{n1["fails_from_room"]} and {l1["fails_from_room"]} of them from the same room). In both seed pairs LBS ends lower, by {d0} and {d1} queries of 240. That is the same direction twice and small: '
                          f'within one run the count moves by up to {swing} queries between saved epochs 1,000 apart. Two pairs do not settle a difference of this size; they do rule out a large loss from the segment loss on this house, '
                          'and they show no gain from it on this house either.</li>')
                l2, n2 = R.get(('lbs', 2, True)), R.get(('noS', 2, True))
                if l2 and n2:
                    second += (f'<li><b>Third training seed: no segment {f(n2["solved"])} %, LBS {f(l2["solved"])} %</b> ({n2["fails_per_planning_seed"][0]} and {l2["fails_per_planning_seed"][0]} failed of 240 on planning seed 0; '
                               f'{n2["fails_from_room"]} and {l2["fails_from_room"]} of them from the same room). Three pairs, LBS lower each time, by 1.7, 2.5 and '
                               f'{n2["solved"] - l2["solved"]:.1f} points: a small cost of the segment loss on this house is the consistent reading, and it is smaller than the movement between saved epochs.</li>')
                elif l2:
                    second += f'<li>Third training seed: LBS {f(l2["solved"])} %; its no-segment partner is training.</li>'
            else:
                second = ('<li><b>What this does not show:</b> that the segment loss helps here. With it the count is 4 queries lower in this seed pair; one pair cannot tell 4 queries from chance. '
                          'The second training seed of both arms is training.</li>')
            A('<ul>'
              f'<li><b>The gap to 100 % is not the segment loss.</b> Without segment, the same code and seed give {f(n0["solved"])} %; with it {f(l0["solved"])} %. The difference is {l0["fails_per_planning_seed"][0] - n0["fails_per_planning_seed"][0]} '
              f'of 240 queries; {n0["shared_with_same_seed_no_segment"] and l0["shared_with_same_seed_no_segment"]} failed queries are the same in both.</li>'
              f'<li><b>It is one room.</b> {AL["queries_from_room"]} of the 240 queries start in the room at the top left of the house ({AL["long_from_room"]} of the {AL["long_queries"]} longest). '
              f'{n0["fails_from_room"]} of the {n0["fails_per_planning_seed"][0]} failures without segment and {l0["fails_from_room"]} of the {l0["fails_per_planning_seed"][0]} with it start there, on every planning seed. In each run the plans stop at one spot just outside it, '
              f'({n0["stall_end_median"][0]:.1f}, {n0["stall_end_median"][1]:.1f}) cells without segment and ({l0["stall_end_median"][0]:.1f}, {l0["stall_end_median"][1]:.1f}) with it: the field has a local minimum on the way out of that room.</li>'
              f'<li><b>The room opens and closes during training, in every run.</b> Both runs of this benchmark solve {240 - ep(l0, 4000)["fails"]} of 240 at epoch 4,000 and fall back at 5,000. The earlier study’s run failed on the same room at '
              f'epochs 1,000–3,000 ({", ".join(str(ep(old, e)["fails_from_room"]) for e in (1000, 2000, 3000))} failures from it) and had it open at 5,000. So on this house the score at the last epoch moves by about 6 points with '
              'the state of one room, whichever arm is trained.</li>'
              + second + '</ul>')
            rows = [[NAME[r['arm']] + ('' if r['source'] == 'jeffrey/lbs' else ' (earlier study’s run)'), r['seed'], f(r['solved']), r['fails_per_planning_seed'][0], r['fails_from_room'],
                     ' / '.join(str(240 - c['fails']) for c in r['course']), f'<code>{r["id"]}</code>'] for r in AL['runs']]
            A(details('Allensville runs: the numbers', table(['arm', 'training seed', 'solved % (3 planning seeds)', 'failed of 240 (planning seed 0)', 'of them from the room', 'solved of 240 at the saved epochs', 'registered run'], rows, left=1)
                      + '<p>The room: starts with x &lt; −2 and y &gt; 11 cells (box centre at 0), chosen from the failures of the same-seed pair. <code>jeffrey/lbs/allens.py</code>.</p>'))

    for s in ('Lakeville',):
        if f'pair_{s}' in C:
            l, n_ = [r for r in D.sel(s, 'lbs') if r['source'] == 'jeffrey/lbs'][0], [r for r in D.sel(s, 'noS') if r['source'] == 'jeffrey/lbs'][0]
            old = [r for r in D.extra if r['scene'] == s and r['arm'] == 'noS']
            fl = lambda r: [c['runs'] - c['success'] for c in r['course']]  # noqa: E731
            A(f'<h3>{s}: the matched pair</h3>')
            A(f'<p>No segment, this benchmark’s code, the same seed and GPU as the LBS run: <b>{f(n_["solved"])} %</b>; LBS <b>{f(l["solved"])} %</b>'
              + (f' (the reused run of the earlier study: {f(old[0]["solved"])} %)' if old else '') + f'. Failed queries of {l["course"][0]["runs"]} at epochs 1,000 to 5,000 (planning seed 0): '
              f'no segment {" / ".join(map(str, fl(n_)))}, LBS {" / ".join(map(str, fl(l)))}. The two runs are identical through epoch 2,000.</p>')
            A(ch(f'pair_{s}'))
    hp = [s for s in ('Lakeville',) if f'pair_{s}' in C]
    if hp:
        def own1(s, a):
            return [r for r in D.sel(s, a) if r['source'] == 'jeffrey/lbs' and r['seed'] == 87358][0]
        parts = []
        for s in hp:
            l, n_ = own1(s, 'lbs'), own1(s, 'noS')
            late = [(cl['runs'] - cl['success'], cn['runs'] - cn['success']) for cl, cn in zip(l['course'], n_['course']) if cl['epoch'] > 2000]
            parts.append(f'<b>{s}: {l["solved"] - n_["solved"]:+.1f} points</b> for LBS (failed queries after the segment loss starts, LBS / no segment: ' + ', '.join(f'{a_} / {b_}' for a_, b_ in late) + ')')
        A('<p><b>Reading the houses together.</b> ' + '; '.join(parts).replace('-', '−') + '; Allensville ' + ', '.join(f'{l_["solved"] - n_["solved"]:+.1f}' for sd in (87358, 1, 2) for l_ in [r for r in D.sel('Allensville', 'lbs') if r['seed'] == sd and r['source'] == 'jeffrey/lbs'][:1] for n_ in [r for r in D.sel('Allensville', 'noS') if r['seed'] == sd and r['source'] == 'jeffrey/lbs'][:1]).replace('-', '−') + ' points in its seed pairs. '
          + ('The sign is not the same from house to house: segment costs about 6 points on Lakeville, where the runs separate as soon as it starts and stay apart, and gains about as much on Klickitat, where the run '
             'without it swings by up to 25 queries between saved epochs. ' if len(hp) == 2 else '')
          + (f'Klickitat, taken out of the benchmark on 2026-11-20, went the other way in its matched pair: {own1("Klickitat", "lbs")["solved"] - own1("Klickitat", "noS")["solved"]:+.1f} points for LBS. '
             if D.sel('Klickitat', 'lbs') and [r for r in D.sel('Klickitat', 'noS') if r['source'] == 'jeffrey/lbs'] else '')
          + 'So the matched pairs do not show that the segment loss is harmful on the houses, and they do not show that it helps: with one seed pair per house its effect there is within about ±6 points and of '
          'either sign, the same size as the movement between saved epochs of a single run. More seeds per house would be needed to say more; the cause of the Lakeville cost is not tested.</p>')

    # ---------------------------------------------------------------- lift on the houses
    HS = FAMILIES[2][2]
    nl, nt, lb = {s: D.one(s, 'noL') for s in HS}, {s: D.one(s, 'ntrl') for s in HS}, {s: D.one(s, 'lbs') for s in HS}
    if all(nl.values()) and all(nt.values()) and all(lb.values()):
        bad = [s for s in HS if nl[s]['solved'] < nt[s]['solved']]
        good = [s for s in HS if s not in bad]
        A('<h2 id="lift-houses">The houses: is lift needed?</h2>')
        if bad:
            ratios = lambda r: ' / '.join(f(r['by_distance'].get(b), 2) for b in BINS if r['by_distance'].get(b) is not None)  # noqa: E731
            far = lambda r: [r['by_distance'][b] for b in BINS if r['by_distance'].get(b) is not None][-1]  # noqa: E731
            cb = lambda r, e, bins: sum(next(c for c in r['course_bins'] if c['epoch'] == e).get(b, 0) for b in bins)  # noqa: E731
            A('<p>The arm without lift is Bellman + segment on ntrl-demo’s own input. '
              + (f'On {" and ".join(good)} it solves {" and ".join(f(nl[s]["solved"]) for s in good)} %, above LBS. ' if good else '')
              + f'On {" and ".join(bad)} it ends at <b>{" and ".join(f(nl[s]["solved"]) for s in bad)} %</b>, below ntrl-demo itself ({" and ".join(f(nt[s]["solved"]) for s in bad)} %), which has neither part '
              f'(LBS: {" and ".join(f(lb[s]["solved"]) for s in bad)} %). Two parts that help on every maze make these two houses worse when lift is missing, so this needs a mechanism.</p>')
            A(grid(*[f'far_{s}' for s in bad]))
            A(grid(*[f'far_{s}' for s in good]))
            A('<ul>')
            A('<li><b>The failures are stalls, not walls.</b> ' + '; '.join(f'{s}: {f(nl[s]["stall"])} % of the runs stall and {f(nl[s]["wall"])} % go through a wall' for s in bad)
              + '. On the mazes the same arm failed through walls.</li>')
            A('<li><b>The travel times stop growing with distance.</b> ' + ' '.join(
                f'On {s} the median of learned over true time is {ratios(nl[s])} at {", ".join(b.replace("-", "–") for b in BINS if nl[s]["by_distance"].get(b) is not None)} cells; '
                f'the largest learned time is {f(nl[s]["t_max"], 2)} where the true one reaches {f(nl[s]["t_star_max"], 2)}. '
                f'Queries solved by true path length ({", ".join(b.replace("-", "–") for b in QB if b in nl[s]["by_bin"])} cells): {" / ".join(f(nl[s]["by_bin"][b], 0) for b in QB if b in nl[s]["by_bin"])} %.' for s in bad)
              + ' The field is right near the goal and flat beyond, so a far goal gives the planner no slope to follow.</li>')
            A('<li><b>ntrl-demo alone and LBS keep the far field.</b> In the farthest bin: ' + '; '.join(f'{s}: ntrl-demo {f(far(nt[s]), 2)}, LBS {f(far(lb[s]), 2)}, no lift {f(far(nl[s]), 2)}' for s in bad)
              + '. So the flat far field is not a limit of the plain input and not a property of the Bellman term as such: it appears when the Bellman term runs without lift on the larger scene.</li>')
            if good and all(r.get('course_bins') for r in nl.values()):
                g = good[0]
                A('<li><b>It is there before the segment loss starts</b> (epoch 2,000), so segment is not the cause. Queries longer than 16 cells solved at epochs 1,000 to 5,000 (planning seed 0): '
                  + '; '.join(f'{s} ' + ' / '.join(str(cb(nl[s], e, QB[4:])) for e in (1000, 2000, 3000, 4000, 5000)) + f' of {40 * sum(1 for b in QB[4:] if b in nl[s]["by_bin"])}' for s in bad + [g])
                  + f'. On {g} the arm also starts with nothing beyond 16 cells and then builds the far field outward, one distance bin after the other; on {" and ".join(bad)} that build-up does not happen within 5,000 epochs.</li>')
            A('<li><b>What differs between the houses</b> is their size in cells and, with it, the weight of the Bellman term, which is set as (0.03 / step radius in box units)² and so grows with the square of the scene size: '
              + ', '.join(f'{s} {f(nl[s]["bell"])}' for s in HS) + '. Whether that weight is the cause is what the control below tests.</li>')
            own_nt = [r for r in D.sel('Lakeville', 'ntrl') if r['source'] == 'jeffrey/lbs']
            ctl = [r for r in D.controls if r['arm'] == 'noLw']
            wait = [n for n, have in (('ntrl-demo trained by this benchmark’s code on Lakeville, with its success during training', own_nt), ('no lift on Lakeville with the Bellman weight of Allensville (× 0.27)', ctl)) if not have]
            if own_nt:
                o = own_nt[0]
                A(f'<li><b>Control: ntrl-demo trained by this benchmark’s code on Lakeville</b> (same seed; no Bellman, no segment, no lift) solves {f(o["solved"])} % and has the far field from the start. '
                  'Queries longer than 16 cells solved at epochs 1,000 to 5,000: ntrl-demo ' + ' / '.join(str(cb(o, e, QB[4:])) for e in (1000, 2000, 3000, 4000, 5000))
                  + ', no lift ' + ' / '.join(str(cb(nl['Lakeville'], e, QB[4:])) for e in (1000, 2000, 3000, 4000, 5000)) + ' of 120. '
                  'So with the plain input the Bellman term holds the far field down from the first thousand epochs on; the plain input does not take long to learn it.</li>')
            if ctl and 'Lakeville' in nl:
                c_ = ctl[0]
                A(f'<li><b>Control: no lift on Lakeville with the Bellman weight cut to Allensville’s</b> (× 0.27, weight {f(c_["bell"])}): {f(c_["solved"])} % solved, {f(c_["stall"])} % stalled; '
                  f'at the full weight {f(nl["Lakeville"]["solved"])} % and {f(nl["Lakeville"]["stall"])} %. Queries longer than 16 cells solved at epochs 1,000 to 5,000: '
                  + ' / '.join(str(cb(c_, e, QB[4:])) for e in (1000, 2000, 3000, 4000, 5000)) + f' of 120; median learned / true time {f(c_["ratio"], 2)}. '
                  '<b>So the size of the Bellman weight is not the cause.</b> What is left is the size of the scene in cells itself (83 against 43): why the Bellman term without lift cannot carry the far field across a scene that large is not tested.</li>')
            if wait:
                A('<li><b>Controls still training:</b> ' + '; '.join(wait) + '.</li>')
            A('</ul>')
            rows = []
            for s in HS:
                for c in nl[s].get('course_bins', []):
                    rows.append([s, f'{c["epoch"]:,}'] + [str(c.get(b, '–')) for b in QB])
            A(details('No lift on the houses: queries solved during training by true path length (planning seed 0, of 40 per bin)', table(['house', 'epoch'] + [b.replace('-', '–') + ' cells' for b in QB], rows, left=2)))

    # ---------------------------------------------------------------- how
    A('<h2>How it was run, and what to keep in mind</h2>')
    A('<ul>'
      '<li><b>One training seed per arm</b> (87358). On the 22-cell maze of the pair-lift study the same lift + Bellman recipe moved by 2.5 points between three seeds, and on a 40-cell maze by 10; '
      'differences of a few points between arms on one scene are within that.</li>'
      '<li><b>Success during training</b> is planning seed 0 on the test queries with the network of that epoch, every 500 epochs (1,000 on the houses). The bar charts are the final network with planning seeds 0, 1 and 2.</li>'
      '<li><b>The arms share their start.</b> The segment loss switches on at epoch 2,000, so up to there LBS and “no segment” are the same recipe, and so are “no Bellman” and the lift alone. '
      'They are not the same numbers: the runs went to two different GPUs, and their curves already differ at epoch 500 by that alone. Read differences of a few points before epoch 2,000 as run-to-run variation.</li>'
      '<li><b>Ground truth in the backup.</b> The Bellman arms read the true speed along each candidate step and the true geometry for the collision check (a clearance table of 32 nodes per cell). '
      'The learned-speed variant of the pair-lift page is not used here.</li>'
      '<li><b>Reused runs</b> (Gibson: ntrl-demo, no Bellman, no segment) were trained by the earlier studies’ pipelines with the same loop, data, queries, planner and scorer; their curves during training have fewer points.</li>'
      '<li><b>Training time</b> is what each run took on the GPU it happened to get (RTX 3090 or RTX 5070), one run per GPU.</li>'
      '</ul>')
    rows = []
    for _, _, scenes in FAMILIES:
        for s in scenes:
            for a in ARMS:
                for r in D.sel(s, a):
                    rows.append([TITLE[s], LONG[a], r['seed'], f'{f(r["solved"])} ({r["success"][0]}/{r["success"][1]})', f(r['wall']), f(r['stall']), f(r['within10']), f(r['ratio'], 3), f(r['seen_through']),
                                 f(r['train_min'], 0), f'<code>{r["id"]}</code>', r['source']])
    A(details('Every run: the numbers', table(['scene', 'arm', 'seed', 'solved %', 'through a wall %', 'stalled %', 'T within 10 %', 'median T / T*', 'detours seen through %', 'training min', 'registered run', 'from'], rows, left=2)))
    rows = []
    for _, _, scenes in FAMILIES:
        for s in scenes:
            for a in ARMS:
                r = D.one(s, a)
                if r and r.get('by_bin'):
                    rows.append([TITLE[s], NAME[a]] + [f(r['by_bin'].get(b), 0) for b in ('0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128')])
    A(details('Queries solved by true path length (cells)', table(['scene', 'arm', '0.5–2', '2–4', '4–8', '8–16', '16–32', '32–64', '64–128'], rows, left=2)))
    rows = []
    for _, _, scenes in FAMILIES:
        for s in scenes:
            for a in ARMS:
                r = D.one(s, a)
                if r and r.get('course'):
                    rows.append([TITLE[s], NAME[a]] + [next((f(c['pct'], 0) for c in r['course'] if c['epoch'] == e), '–') for e in range(500, 5001, 500)])
    A(details('Queries solved during training (planning seed 0), by epoch', table(['scene', 'arm'] + [f'{e:,}' for e in range(500, 5001, 500)], rows, left=2)))
    A('<footer>jeffrey/lbs · report.py → report/runs.json · page.py · built from the registered runs listed above.</footer>')
    return '\n'.join(H)


def main():
    D = Data()
    C = build_charts(D)
    css = (LS / 'page_assets/style.css').read_text()
    js = (LS / 'page_assets/charts.js').read_text()
    page = (f'<title>{TITLE_TAG}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">\n'
            f'<style>\n{css}\n{EXTRA_CSS}</style>\n<div class="wrap">\n{body(D, C)}\n</div>\n<div class="tip" id="tip" hidden></div>\n'
            f'<script id="data" type="application/json">{json.dumps(dict(charts=C))}</script>\n<script>\n{js}\n</script>\n')
    (OUT / 'lbs.html').write_text(page)
    print(f'{OUT / "lbs.html"}: {len(page) / 1024:.0f} KB, {len(C)} charts')


if __name__ == '__main__':
    main()
