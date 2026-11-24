"""Findings page of the arm-placement study: report/arm_placement.html from report/summary.json (report.py), report/demo.json (the
demo run's results.json, copied by finalize.py) and report/notes.json (hand-written numbers: throughput measurements).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/page.py

Chart code and style sheet are jeffrey/lift_scale/page_assets (the repository's report pages share them).
Published as the artifact "2026-11-23 Two-Arm Placement".
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

TITLE_TAG = '2026-11-23 Two-Arm Placement'
TASKS = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']
TASK = dict(free_to_free='free → free', free_to_compartment='free → zone', compartment_to_free='zone → free',
            compartment_to_compartment='zone → another zone', same_compartment='inside one zone', anywhere='anywhere → anywhere')
EXTRA_CSS = '''
.sumtbl { overflow-x: auto; margin: 10px 0 16px; }
.sumtbl table { min-width: 0; width: 100%; max-width: 900px; font-size: 14px; }
.sumtbl td, .sumtbl th { padding: 6px 9px; white-space: normal; }
.scene { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px 8px; margin: 14px 0 6px; }
.scene svg { display: block; width: 100%; height: auto; max-width: 760px; }
.scene .ctitle { font-weight: 600; font-size: 14.5px; margin: 0 0 2px; }
.scene .csub { font-size: 13px; color: var(--ink-2); margin: 0 0 8px; max-width: 100ch; }
.demos { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 330px), 1fr)); gap: 14px; margin: 14px 0; }
.demo { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 12px; min-width: 0; }
.demo h4 { font-family: var(--font-head); font-size: 15.5px; margin: 0 0 2px; }
.demo .meta { font-size: 12.5px; color: var(--ink-2); margin: 0 0 6px; }
.demo svg { display: block; width: 100%; height: auto; }
.demo .views { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.demo .cap { font-size: 11.5px; color: var(--ink-3); text-align: center; }
pre.cfg { font-family: var(--font-mono); font-size: 12px; background: var(--code-bg); border-radius: 4px; padding: 10px 12px; overflow-x: auto; margin: 6px 0 10px; }
'''


def pct(a, b):
    return 100 * a / b if b else None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


class Data:
    def __init__(self):
        d = json.loads((OUT / 'summary.json').read_text())
        self.P = d['placements']
        self.pending = d['pending']
        self.reach = d['reach_grid']
        self.plans = d.get('plans', [])
        self.demo = json.loads((OUT / 'demo.json').read_text()) if (OUT / 'demo.json').exists() else None
        self.notes = json.loads((OUT / 'notes.json').read_text()) if (OUT / 'notes.json').exists() else {}
        self.depth = json.loads((OUT / 'depth.json').read_text()) if (OUT / 'depth.json').exists() else {}

    def by_name(self, name):
        return next((p for p in self.P if p['name'] == name), None)

    def fields(self, p, label):
        return sorted([x for x in p['fields'] if x['label'] == label], key=lambda x: x['seed'])

    def fwd(self, p, label, task=None):
        """Mean over training seeds of the forward planner's solved share (%), all queries or one stratum."""
        fs = self.fields(p, label)
        return mean([x['solved'] if task is None else pct(x['by_task'][task], x['per_task_runs']) for x in fs])

    def plan(self, p, label, kind, steps, task=None):
        out = []
        for x in self.fields(p, label):
            for q in x.get('planners', []):
                if q['kind'] == kind and q['steps'] == steps and q['samples'] == 50:
                    out.append(q['solved'] if task is None else pct(q['by_task'][task], x['per_task_runs']))
        return mean(out)

    def line_share(self, p, task=None):
        pr = p.get('prepare')
        if not pr:
            return None
        return pct(pr['straight_line'], pr['queries']) if task is None else pct(pr['straight_line_by_task'][task], pr['queries'] / 6)

    def euclid(self, p, task=None):
        e = p.get('euclid')
        if not e:
            return None
        return e['solved'] if task is None else pct(e['by_task'][task], e['success'][1] / 6)


def pname(p):
    return f'{p["dx"]:.2f} m' + (f', spacing ±{p["s"]:.2f}' if abs(p['s'] - 0.30) > 1e-6 else '') + (f', height {p["h"]:.2f}' if abs(p['h'] - 0.60) > 1e-6 else '')


def scene_svg(P, best=None):
    """Side view (x-z, cupboard frame): the cupboard with its three zones and the base positions that were tested."""
    W, H = 760, 330
    sx = 300.0                     # px per metre
    x0, zf = 420.0, 300.0          # cupboard front face x; floor z (px)
    X = lambda x: x0 + x * sx      # noqa: E731  (x in metres from the front face; negative = towards the robot)
    Z = lambda z: zf - z * sx * 0.85 / 1.0 * 0.5      # noqa: E731
    Z = lambda z: zf - z * 150.0   # noqa: E731,F811  (1.82 m cupboard -> 273 px)
    o = [f'<svg viewBox="0 0 {W} {H + 42}" role="img" aria-label="Side view of the cupboard and the tested base positions">',
         f'<defs><clipPath id="above"><rect x="0" y="0" width="{W}" height="{zf}"/></clipPath></defs>']
    o.append(f'<line x1="20" x2="{W - 10}" y1="{zf}" y2="{zf}" stroke="var(--ink-3)" stroke-width="1.5"/>')
    panel = 'fill="var(--ink-3)" opacity="0.55"'
    for (xa, xb, za, zb) in [(0, 0.56, 0, 0.08), (0, 0.56, 1.74, 1.82), (0.53, 0.56, 0.08, 1.74), (0, 0.53, 0.6133, 0.6433), (0, 0.53, 1.1767, 1.2067)]:
        o.append(f'<rect x="{X(xa):.1f}" y="{Z(zb):.1f}" width="{(xb - xa) * sx:.1f}" height="{(zb - za) * 150:.1f}" {panel}/>')
    for name, zc in (('bottom zone', 0.347), ('middle zone', 0.91), ('top zone', 1.473)):
        o.append(f'<text x="{X(0.265):.1f}" y="{Z(zc) + 4:.1f}" text-anchor="middle" font-size="12" fill="var(--ink-2)">{name}</text>')
    o.append(f'<text x="{X(0.28):.1f}" y="{Z(1.82) - 6:.1f}" text-anchor="middle" font-size="11.5" fill="var(--ink-3)">cupboard, 0.56 m deep, opens to the left</text>')
    seen = {}
    for p in P:
        if p['name'] == 'base' or not p['fields']:
            continue
        key = (round(p['dx'], 2), round(p['h'], 2))
        seen.setdefault(key, []).append(p)
    row = {dx: i % 4 for i, dx in enumerate(sorted({k[0] for k in seen}))}
    labelled = set()
    for (dx, h), ps in sorted(seen.items()):
        is_best = best is not None and any(q['name'] == best for q in ps)
        col = 'var(--s2)' if is_best else 'var(--s1)'
        xb = X(-dx)
        o.append(f'<rect x="{xb - 4:.1f}" y="{Z(h - 0.04):.1f}" width="8" height="{(h - 0.04) * 150:.1f}" fill="{col}" opacity="{0.9 if is_best else 0.35}"/>')
        o.append(f'<circle cx="{xb:.1f}" cy="{Z(h + 0.1625):.1f}" r="{5 if is_best else 3.5}" fill="{col}"/>')
        if dx not in labelled:
            labelled.add(dx)
            o.append(f'<line x1="{xb:.1f}" x2="{xb:.1f}" y1="{zf + 2:.1f}" y2="{zf + 5 + 13 * row[dx]:.1f}" stroke="var(--ink-3)" stroke-width="1"/>')
            o.append(f'<text x="{xb:.1f}" y="{zf + 16 + 13 * row[dx]:.1f}" text-anchor="middle" font-size="11" fill="var(--ink-2)" font-family="var(--font-mono)">{dx:.2f}</text>')
        if is_best:
            r = 0.85 * 150
            o.append(f'<ellipse cx="{xb:.1f}" cy="{Z(h + 0.1625):.1f}" rx="{0.85 * sx:.1f}" ry="{r:.1f}" fill="none" stroke="{col}" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.8" clip-path="url(#above)"/>')
    o.append(f'<text x="{X(0.02):.1f}" y="{zf + 16:.1f}" font-size="11" fill="var(--ink-3)">distance of the bases from the front face (m)</text>')
    o.append('</svg>')
    return '\n'.join(o)


def demo_views(D, item, field):
    """Front (y-z) and top (x-y) views of one picked query: start and goal skeletons of both arms and the grasp-point paths."""
    sc = D.demo['scene']
    plan = item['plans'][field]
    sk, gp = plan['skeleton'], plan['grasp']
    cub = [c for c in sc['cuboids'] if c['name'].startswith('shelf')]
    out = []
    for view, (ia, ib, flip) in (('front view (from the robots)', (1, 2, True)), ('top view', (1, 0, True))):
        xs = [c['pose'][ia] - c['dims'][ia] / 2 for c in cub] + [c['pose'][ia] + c['dims'][ia] / 2 for c in cub] + [b['base'][ia] for b in sc['robots']]
        ys = [c['pose'][ib] - c['dims'][ib] / 2 for c in cub] + [c['pose'][ib] + c['dims'][ib] / 2 for c in cub] + [b['base'][ib] for b in sc['robots']]
        for fr in sk:
            for arm in fr:
                xs += [pt[ia] for pt in arm]
                ys += [pt[ib] for pt in arm]
        xa, xb, ya, yb = min(xs) - 0.05, max(xs) + 0.05, min(ys) - 0.05, max(ys) + 0.05
        W = 300.0
        s = W / (xb - xa)
        Hh = (yb - ya) * s
        X = (lambda v: (xb - v) * s) if flip else (lambda v: (v - xa) * s)      # y to the left in both views (seen from the robots / from above with x up)
        Y = lambda v: (yb - v) * s      # noqa: E731
        o = [f'<svg viewBox="0 0 {W:.0f} {Hh:.0f}" role="img" aria-label="{view} of query {item["query"]}">']
        for c in cub:
            if view.startswith('front') and c['name'] == 'shelf_back':
                continue
            o.append(f'<rect x="{min(X(c["pose"][ia] - c["dims"][ia] / 2), X(c["pose"][ia] + c["dims"][ia] / 2)):.1f}" y="{Y(c["pose"][ib] + c["dims"][ib] / 2):.1f}" width="{c["dims"][ia] * s:.1f}" height="{c["dims"][ib] * s:.1f}" fill="var(--ink-3)" opacity="0.28"/>')
        for a, col in ((0, 'var(--s1)'), (1, 'var(--s2)')):
            path = ' '.join(f'{"M" if i == 0 else "L"}{X(g[a][ia]):.1f},{Y(g[a][ib]):.1f}' for i, g in enumerate(gp))
            o.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="1.6" opacity="0.9"/>')
            for fr, op, dash in ((sk[0], 0.95, ''), (sk[-1], 0.95, ' stroke-dasharray="4 3"')):
                pts = ' '.join(f'{X(pt[ia]):.1f},{Y(pt[ib]):.1f}' for pt in fr[a])
                o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity="{op}"{dash}/>')
            o.append(f'<circle cx="{X(sk[0][a][7][ia]):.1f}" cy="{Y(sk[0][a][7][ib]):.1f}" r="4" fill="{col}" stroke="var(--surface)" stroke-width="1.5"/>')
            o.append(f'<circle cx="{X(sk[-1][a][7][ia]):.1f}" cy="{Y(sk[-1][a][7][ib]):.1f}" r="4" fill="var(--surface)" stroke="{col}" stroke-width="2"/>')
        o.append('</svg>')
        out.append(f'<div>{"".join(o)}<div class="cap">{view}</div></div>')
    return '<div class="views">' + ''.join(out) + '</div>'


def main():
    D = Data()
    C, H = {}, []
    A = H.append
    ch = lambda k: chart(k) if k in C else ''  # noqa: E731
    sweep = sorted([p for p in D.P if p['name'] != 'base' and abs(p['s'] - 0.30) < 1e-6 and abs(p['h'] - 0.60) < 1e-6 and D.fields(p, 'lift')], key=lambda p: p['dx'])
    base = D.by_name('base')
    N = D.notes
    best = N.get('best')
    bp = D.by_name(best) if best else None
    BK, BS = N.get('planner_kind', 'bidir'), N.get('planner_steps', 600)

    # ---- charts
    if sweep:
        xs = [p['dx'] for p in sweep]
        series = [('lift field, two-ended planner', 2, [(p['dx'], D.plan(p, 'lift', BK, BS)) for p in sweep], ''),
                  ('lift field, ntrl-demo planner', 1, [(p['dx'], D.fwd(p, 'lift')) for p in sweep], ''),
                  ('ntrl-demo field, two-ended planner', 6, [(p['dx'], D.plan(p, 'ntrl', BK, BS)) for p in sweep], ''),
                  ('ntrl-demo field, ntrl-demo planner', 3, [(p['dx'], D.fwd(p, 'ntrl')) for p in sweep], ''),
                  ('control: no field (joint distance), ntrl-demo planner', 4, [(p['dx'], D.euclid(p)) for p in sweep], '5 4'),
                  ('control: straight line', 5, [(p['dx'], D.line_share(p)) for p in sweep], '2 4')]
        series = [s for s in series if any(v is not None for _, v in s[2])]
        C['sweep'] = line('Queries solved against the distance of the arms from the cupboard', 'Share of planning runs that reach the goal without any collision (300 certified queries per placement × 3 planning seeds; mean of the training seeds). Bases ±0.30 m from the centre plane, 0.60 m above the floor.',
                          series, xlabel='distance from the bases to the cupboard front (m)', ylabel='solved (%)', xticks=xs, yticks=[0, 20, 40, 60, 80, 100], ymin=0, ymax=100, digits=1, xname='distance', height=340)
        for key, title, fn in (('tasks_fwd', 'Lift field with ntrl-demo’s planner, by kind of query', lambda p, t: D.fwd(p, 'lift', t)),
                               ('tasks_bi', 'Lift field with the two-ended planner, by kind of query', lambda p, t: D.plan(p, 'lift', BK, BS, t))):
            ser = [(TASK[t], i + 1, [(p['dx'], fn(p, t)) for p in sweep], '') for i, t in enumerate(TASKS)]
            if any(v is not None for s_ in ser for _, v in s_[2]):
                C[key] = line(title, '150 runs per point and training seed (50 queries × 3 planning seeds). "Zone" = a compartment of the cupboard; both arms obey the rule (e.g. zone → another zone: each arm starts in one zone and ends in a different one).',
                              ser, xlabel='distance from the bases to the cupboard front (m)', ylabel='solved (%)', xticks=xs, yticks=[0, 20, 40, 60, 80, 100], ymin=0, ymax=100, digits=1, xname='distance', height=320)
        ease = [('reach: zone volume at least one arm reaches', 1, [(p['dx'], 100 * p['reach']['union_mean']) for p in sweep if p.get('reach')], ''),
                ('reach: zone volume both arms reach', 5, [(p['dx'], 100 * p['reach']['both_mean']) for p in sweep if p.get('reach')], ''),
                ('configurations in the task box that are collision-free', 3, [(p['dx'], 100 * p['prepare']['valid_of_in_box']) for p in sweep if p.get('prepare')], ''),
                ('of those, at full speed (5 cm from everything)', 4, [(p['dx'], 100 * p['prepare']['full_speed_of_valid']) for p in sweep if p.get('prepare')], ''),
                ('candidate queries the witness planner certifies (first round)', 2, [(p['dx'], pct(sum(v[0] for v in p['tests']['first_round'].values()), sum(v[1] for v in p['tests']['first_round'].values()))) for p in sweep if p.get('tests')], '')]
        C['ease'] = line('What moving back changes, without any learned field', 'Reach falls; the free space opens and the witness planner (straight line, then RRT-Connect) certifies more of the random candidate queries.',
                         ease, xlabel='distance from the bases to the cupboard front (m)', ylabel='%', xticks=xs, yticks=[0, 20, 40, 60, 80, 100], ymin=0, ymax=100, digits=1, xname='distance', height=320)
    if sweep and D.depth:
        dd = [(p['dx'], D.depth.get(p['name'])) for p in sweep]
        ser = [(nm, slot, [(dx, 100 * d['zone_to_zone'][k]) for dx, d in dd if d], '') for k, nm, slot in (('max', 'deepest query', 2), ('p90', '90th percentile', 4), ('median', 'median', 1))]
        C['depth'] = line('How far inside the cupboard the hands work', 'Depth of the grasp point behind the front face at the starts and goals of the zone → another zone test queries (50 queries, both arms, both ends). The shelves are 53 cm deep.',
                          ser, xlabel='distance from the bases to the cupboard front (m)', ylabel='depth (cm)', xticks=xs, yticks=[0, 10, 20, 30, 40, 50], ymin=0, ymax=55, digits=1, xname='distance', height=300)
        labels = []
        for _, d in dd:
            for b in (d or {}).get('by_depth', {}).get('bins', []):
                if b['label'] not in labels:
                    labels.append(b['label'])
        ser = []
        for i, lab in enumerate(labels):
            pts = []
            for dx, d in dd:
                b = next((b for b in (d or {}).get('by_depth', {}).get('bins', []) if b['label'] == lab), None)
                if b and b['runs'] >= 30:
                    pts.append((dx, pct(b['solved'], b['runs'])))
            if pts:
                ser.append((lab, i + 1, pts, ''))
        if ser:
            C['bydepth'] = line('Control: the same depth at different distances', f'Lift field, two-ended planner ({BS:,} steps per end). Each query is filed by its deepest hand at either end; points with fewer than 30 planning runs are left out.',
                                ser, xlabel='distance from the bases to the cupboard front (m)', ylabel='solved (%)', xticks=xs, yticks=[0, 20, 40, 60, 80, 100], ymin=0, ymax=100, digits=1, xname='distance', height=300)
    var = [p for p in D.P if p['name'] != 'base' and D.fields(p, 'lift')]
    dxs = sorted({p['dx'] for p in var if sum(1 for q in var if q['dx'] == p['dx']) > 1})
    if dxs:
        groups = []
        for dx in dxs:
            rows = []
            for p in sorted([q for q in var if q['dx'] == dx], key=lambda q: (q['h'], q['s'])):
                nm = f'spacing ±{p["s"]:.2f} m, height {p["h"]:.2f} m'
                rows.append((nm + ' · two-ended', D.plan(p, 'lift', BK, BS), 2, ''))
                rows.append((nm + ' · ntrl-demo planner', D.fwd(p, 'lift'), 1, ''))
            groups.append((f'{dx:.2f} m from the cupboard', rows))
        C['variants'] = hbar('Spacing and height of the bases (lift field)', 'Solved share of all runs, mean of the training seeds. The spacing is the distance of each base from the cupboard’s centre plane.', groups, value_name='solved',
                             legend=[('two-ended planner', 2), ('ntrl-demo planner', 1)])
    best_p = bp
    bp = D.by_name(N.get('planner_pl') or best) or bp        # the planner comparison and the training curve are shown for this placement
    if bp:
        groups = []
        kinds = [('forward', 600, 'ntrl-demo planner, start → goal (600 steps)', 1), ('reverse', 600, 'the same planner, goal → start (600 steps)', 3), ('bidir', 300, 'two-ended, 300 steps per end', 4), ('bidir', 600, 'two-ended, 600 steps per end', 2),
                 ('hub', 300, 'via a retracted pose, 300 steps per leg', 5), ('hub', 600, 'via a retracted pose, 600 steps per leg', 6)]
        for t in [None] + TASKS:
            rows = []
            for kind, steps, nm, slot in kinds:
                v = D.fwd(bp, 'lift', t) if kind == 'forward' else D.plan(bp, 'lift', kind, steps, t)
                if v is not None:
                    rows.append((nm, v, slot, ''))
            groups.append(('all queries' if t is None else TASK[t], rows))
        C['planners'] = hbar(f'One field, several planners ({pname(bp)})', 'The same trained lift field. Only the way the planner uses it changes; none of them uses a collision checker.', groups, value_name='solved')
        ser = []
        for pl_, lab, kind, nm, slot, dash in ((bp, 'lift', 'bidir', f'lift field, two-ended, {pname(bp)}', 2, ''), (bp, 'lift', 'forward', f'lift field, ntrl-demo planner, {pname(bp)}', 1, ''),
                                               (base, 'lift', 'bidir', 'lift field, two-ended, 0.25 m (earlier suite)', 4, ''), (base, 'lift', 'forward', 'lift field, ntrl-demo planner, 0.25 m (earlier suite)', 5, ''),
                                               (bp, 'ntrl', 'bidir', f'ntrl-demo field, two-ended, {pname(bp)}', 3, '')):
            if pl_ is None:
                continue
            mult = 2 if kind == 'bidir' else 1
            pts = [(mult * stp, (D.fwd(pl_, lab) if kind == 'forward' and stp == 600 else D.plan(pl_, lab, kind, stp))) for stp in (300, 600, 1200, 2400, 4800)]
            pts = [(x_, v) for x_, v in pts if v is not None]
            if pts:
                ser.append((nm, slot, pts, dash))
        if ser:
            C['budget'] = line('Solved against the number of planner moves', 'Moves = MPPI steps taken in total (the two-ended planner spends half at each end). Mean of the training seeds that were evaluated at that budget.', ser,
                               xlabel='planner moves per query', ylabel='solved (%)', xticks=[600, 1200, 2400, 4800, 9600], yticks=[0, 20, 40, 60, 80, 100], ymin=0, ymax=100, digits=1, xname='moves', height=320)
        ser = []
        for lab, nm, slot in (('lift', 'lift field', 1), ('ntrl', 'ntrl-demo field', 3)):
            for x in D.fields(bp, lab)[:1]:
                ser.append((nm, slot, [(e, pct(s_, n)) for e, s_, n, _ in x['validate']], ''))
        if ser:
            C['curve'] = line(f'Solved during training ({pname(bp)}, ntrl-demo planner, planning seed 0)', 'Validation every 500 epochs on the placement’s 300 queries.', ser, xlabel='epoch', ylabel='solved (%)',
                              xticks=[500, 1000, 2000, 3000, 4000, 5000], yticks=[0, 20, 40, 60, 80, 100], ymin=0, ymax=100, digits=1, xname='epoch', height=260)
    if N.get('throughput'):
        t = N['throughput']
        C['throughput'] = line('Training-like jobs sharing one RTX 4090', 'A small double-backward training loop (4,000 points per step) run as 1, 6, 12 and 24 processes on a card that was also generating test suites.',
                               [('steps per second, all processes together', 1, [(r[0], r[0] * r[1]) for r in t], ''), ('steps per second, one process', 4, [(r[0], r[1]) for r in t], '')],
                               xlabel='processes on the card', ylabel='steps / s', xticks=[r[0] for r in t], yticks=[0, 100, 200, 300, 400, 500], ymin=0, ymax=600, digits=0, xname='processes', height=260)

    bp = best_p
    # ---- text
    A('<header class="top"><div class="eyebrow">jeffrey/arm_placement · 2026-11-23 · two UR5e, 12 joints, measured cupboard</div>')
    A('<h1>Where to put two arms in front of the cupboard</h1>')
    A('<p class="note">3-D view of every placement and of the ten configurations: <a href="https://claude.ai/artifact/3Ms5y25wahMnPrqHAw3pUF">arm placement viewer</a>.</p>')
    A(f'<p class="lede">{N.get("lede", "")}</p></header>')
    if N.get('answers'):
        A('<div class="answers">' + ''.join(f'<div><div class="q">{q}</div><div class="a">{a}</div></div>' for q, a in N['answers']) + '</div>')
    if D.pending and N.get('show_pending', True):
        run = [q for q in D.pending if q['status'] == 'running']
        A(f'<p class="note">Status when this page was built: {len(run)} jobs running, {len(D.pending) - len(run)} waiting or not finished. Numbers below cover the finished runs only.</p>')
    A('<h2 id="sweep">Distance from the cupboard</h2>')
    A(N.get('sweep_text', ''))
    A('<div class="scene"><div class="ctitle">The placements tested (side view)</div><div class="csub">Each marker is a base position with its pedestal; the dot is the shoulder. The cupboard stays where it is and the arms move back. '
      + ('The dashed outline is 0.85 m around the shoulder of the chosen placement, roughly where the wrist can go.' if best else '') + '</div>' + scene_svg(D.P, best) + '</div>')
    A(ch('sweep'))
    A(ch('ease'))
    A(N.get('ease_text', ''))
    if C.get('depth'):
        A('<h2 id="depth">The price: how deep the hands get</h2>')
        A(N.get('depth_text', ''))
        A('<div class="grid2">' + ch('depth') + ch('bydepth') + '</div>')
    A('<h2 id="kinds">Which queries get easier</h2>')
    A(N.get('kinds_text', ''))
    A('<div class="grid2">' + ch('tasks_fwd') + ch('tasks_bi') + '</div>')
    A('<h2 id="planner">The two-ended planner</h2>')
    A(N.get('planner_text', ''))
    A(ch('planners'))
    A(ch('budget'))
    A('<h2 id="variants">Spacing and height</h2>')
    A(N.get('variants_text', ''))
    A(ch('variants'))
    if D.demo:
        dm = D.demo
        A('<h2 id="demo">Ten working start and goal configurations</h2>')
        A(N.get('demo_text', ''))
        first = list(dm['fields'])[0]
        cards = []
        for it in dm['picked']:
            pl = it['plans'][first]
            cards.append(f'<div class="demo"><h4>{it["rank"]}. {esc(" / ".join(it["start_zones"]))} → {esc(" / ".join(it["goal_zones"]))}</h4>'
                         f'<p class="meta">left / right arm zone. Joint distance {it["joint_distance_rad"]:.2f} rad, planned path {pl["length_rad"]:.2f} rad; closest approach to the cupboard {1000 * pl["min_obstacle_m"]:.0f} mm, between the arms {1000 * pl["min_inter_arm_m"]:.0f} mm; '
                         f'straight line {"free" if it["straight_line_feasible"] else "blocked"}.</p>' + demo_views(D, it, first) + '</div>')
        A('<div class="demos">' + ''.join(cards) + '</div>')
        A('<p class="note">Blue = left arm, orange = right arm. Solid skeleton and filled dot: start. Dashed skeleton and open dot: goal. Thin line: the grasp point along the planned path.</p>')
        rows = []
        for it in dm['picked']:
            rows.append([str(it['rank']), str(it['query']), esc(TASK.get(it['task'], it['task'])), '<code>' + ', '.join(f'{v:.4f}' for v in it['start_rad']) + '</code>', '<code>' + ', '.join(f'{v:.4f}' for v in it['goal_rad']) + '</code>']
                        + [f'{sum(it["plans"][k]["solved_seeds"])}/3' for k in dm['fields']])
        A(details('The ten configurations in numbers (radians; left arm joints 1–6, then right arm joints 1–6)',
                  table(['#', 'query', 'kind', 'start', 'goal'] + [f'solved, {esc(k)}' for k in dm['fields']], rows, left=3)))
        A(N.get('demo_files', ''))
    A('<h2 id="throughput">Using the two 48 GB cards</h2>')
    A(N.get('throughput_text', ''))
    A(ch('throughput'))
    A('<h2 id="models">Saved models</h2>')
    A(N.get('models_text', ''))
    A('<h2 id="setup">Setup and limits</h2>')
    A(N.get('setup_text', ''))
    A(ch('curve'))

    # ---- tables
    labels = sorted({x['label'] for p in D.P for x in p['fields']})
    rows = []
    for p in D.P:
        rc, pr, ts = p.get('reach'), p.get('prepare'), p.get('tests')
        for x in sorted(p['fields'], key=lambda x: (x['label'], x['seed'])):
            bi = next((q for q in x.get('planners', []) if q['kind'] == BK and q['steps'] == BS), None)
            rows.append([esc(p['name']), f'{p["dx"]:.2f}', f'{p["s"]:.2f}', f'{p["h"]:.2f}', esc(x['label']), str(x['seed']), f'{x["success"][0]} / {x["success"][1]}', f(x['solved']),
                         (f'{bi["success"][0]} / {bi["success"][1]}' if bi else '–'), f(bi['solved']) if bi else '–'] + [str(x['by_task'][t]) + (f' → {bi["by_task"][t]}' if bi else '') for t in TASKS]
                        + [str(x['not_arrived']), str(x['arrived_contact']), f'{x["train_min"]:.0f}', f'<code>{x["run"][-8:]}</code>'])
    A(details('Every field run (ntrl-demo planner → two-ended planner per kind of query)',
              table(['placement', 'distance', 'spacing', 'height', 'field', 'training seed', 'solved, ntrl-demo planner', '%', 'solved, two-ended', '%'] + [TASK[t] for t in TASKS] + ['did not arrive', 'arrived in contact', 'training min', 'run'], rows, left=1)))
    rows = []
    for p in D.P:
        rc, pr, ts, e = p.get('reach'), p.get('prepare'), p.get('tests'), p.get('euclid')
        rows.append([esc(p['name']), f'{p["dx"]:.2f}', f'{p["s"]:.2f}', f'{p["h"]:.2f}',
                     f(100 * rc['union_mean']) if rc else '–', f(100 * rc['both_mean']) if rc else '–', ' / '.join(f'{100 * rc["union"][k]:.0f}' for k in ('bottom', 'middle', 'top')) if rc else '–',
                     f(100 * pr['valid_of_in_box'], 2) if pr else '–', f(100 * pr['full_speed_of_valid']) if pr else '–', f(pr['E']) if pr else '–',
                     f'{pr["straight_line"]} / {pr["queries"]}' if pr else '–', f'{e["success"][0]} / {e["success"][1]}' if e else '–',
                     (f(100 * ts['certified_share']) if ts else 'frozen suite'), (f'{ts["minutes"]:.0f}' if ts else '–'), (f'<code>{ts["run"][-8:]}</code>' if ts else '–'), (f'<code>{pr["run"][-8:]}</code>' if pr else '–')])
    A(details('Every placement: reach, free space, controls, suite generation',
              table(['placement', 'distance', 'spacing', 'height', 'reach, either arm %', 'reach, both arms %', 'either arm: bottom / middle / top %', 'collision-free of in-box %', 'full speed of those %', 'E (cells)',
                     'straight line', 'no field (joint distance)', 'candidates certified %', 'suite minutes', 'suite run', 'prepare run'], rows, left=1)))
    if D.plans:
        rows = [[esc(q['placement'] or ''), esc('zone → zone set' if q.get('suite') == 'hard' else 'six kinds'), esc(q['field']), str(q['seed'] or ''), esc(q['kind']), str(q['steps']), str(q['samples']), f'{q["success"][0]} / {q["success"][1]}', f(q['solved'])] + [str(q['by_task'].get(t, '–')) for t in TASKS]
                + [str(q['not_arrived']), str(q['arrived_contact']), f'<code>{q["run"][-8:]}</code>'] for q in sorted(D.plans, key=lambda q: (q['placement'] or '', q.get('suite') or '', q['field'], q['seed'] or 0, q['kind'], q['steps']))]
        A(details('Every planner evaluation', table(['placement', 'queries', 'field', 'training seed', 'planner', 'steps', 'samples', 'solved', '%'] + [TASK[t] for t in TASKS] + ['did not arrive', 'arrived in contact', 'run'], rows, left=3)))
    A('<footer>Built by <code>jeffrey/arm_placement/page.py</code> from registered runs (<code>report/summary.json</code>). Run ids are the last eight characters of <code>results/registered_&lt;id&gt;</code>.</footer>')

    css = (LS / 'page_assets/style.css').read_text()
    js = (LS / 'page_assets/charts.js').read_text()
    page = (f'<title>{TITLE_TAG}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">\n'
            f'<style>\n{css}\n{EXTRA_CSS}</style>\n<div class="wrap">\n' + '\n'.join(x for x in H if x) + '\n</div>\n<div class="tip" id="tip" hidden></div>\n'
            f'<script id="data" type="application/json">{json.dumps(dict(charts=C))}</script>\n<script>\n{js}\n</script>\n')
    (OUT / 'arm_placement.html').write_text(page)
    print(f'{OUT / "arm_placement.html"}: {len(page) / 1024:.0f} KB, {len(C)} charts: {sorted(C)}')


if __name__ == '__main__':
    main()
