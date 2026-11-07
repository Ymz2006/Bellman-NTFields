"""Report (from registered runs): the field trained on gap x1 planned on the wider gaps (zq --stretch), against the fields
trained on each gap. report/stretch.json, report/stretchview.html (from stretch_template.html).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/stretchreport.py
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))
OUT = HERE / 'report'
S0 = 87358
KS = [1, 2, 4, 8, 12]
BASES = ('c22', 'm10')
BNAME = dict(c22='22×22 cross maze', m10='10×10 maze')
CATS = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']


def role(R):
    """Run id of a gap x1 field -> (base, label, seed)."""
    out = {}
    for s, rid in R['id22']['1'].items():
        out[rid] = ('c22', 'id', int(s))
    for base in BASES:
        for lab in ('id', 'pmzw'):
            for s, e in R['fields'][f'{base}/1'].get(lab, {}).items():
                out[e['_id']] = (base, lab, int(s))
    return out


def collect():
    R = json.loads((OUT / 'zdraw10.json').read_text())
    who = role(R)
    runs = {}                                       # (base, k, kind, lab, seed) -> dict(entry, rows, run)
    for p in sorted((ROOT / 'results').glob('registered_*/results.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get('stage') != 'zq' or d.get('smoke') or not d.get('complete'):
            continue
        for rid, e in d['runs'].items():
            if not e.get('stretch'):
                continue
            info = json.loads((Path(d['queries']) / 'maze.json').read_text())
            base, lab, seed = who[rid]
            runs[(base, info['gap_scale'], e['stretch'], lab, seed)] = dict(plans=e['plans'], rows=p.parent / rid / 'rows.npz', run=p.parent.name.replace('registered_', ''), m=e['quadrant_stretch'])
    return R, runs


def pct(c):
    return None if not c else 100 * c[0] / c[1]


def own(R, base, k, lab, planner, seed=S0):
    """The field trained on gap k: its entry (zdraw10's report)."""
    import zdraw10 as ZD
    if lab == 'id' and base == 'c22' and k == 1:
        return None                                 # its own rows are on maze_scale's 240 queries: use the identity run instead
    return ZD.test_count(R, base, k, lab, planner, seed=seed)


def own_cat(R, base, k, lab, planner, cat, seed=S0):
    import zdraw10 as ZD
    return ZD.test_count(R, base, k, lab, planner, seed=seed, cat=cat)


def zq_index():
    """Training run id -> its rows.npz in a zq run without a stretch (the last one: the plan-warp runs have no x planner, so x is checked by the caller)."""
    out = {}
    for p in sorted((ROOT / 'results').glob('registered_*/results.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get('stage') != 'zq' or d.get('smoke') or not d.get('complete'):
            continue
        for rid, e in d['runs'].items():
            if not e.get('stretch') and e.get('plan_warp', 'own') == 'own' and (p.parent / rid / 'rows.npz').exists():
                out[rid] = p.parent / rid / 'rows.npz'
    return out


def own_rows(R, base, k, lab, idx):
    """(success, time ratio) of the field trained on gap k, plan seeds x queries: id MPPI on the maze (its training run's rows), pmzw MPPI on the warped maze."""
    if lab == 'id':
        rid = R['id22'][str(k)][str(S0)] if base == 'c22' else R['fields'][f'{base}/{k}']['id'][str(S0)]['_id']
        z = np.load(ROOT / 'results' / f'registered_{rid}' / 'rows.npz')
        return z['success'], z['time_ratio']
    rid = R['fields'][f'{base}/{k}']['pmzw'][str(S0)]['_id']
    if rid not in idx:
        return None
    z = np.load(idx[rid])
    return z['z_success'], z['z_time_ratio']


def paired_time(R, runs, base, k, lab, planner, idx):
    """On the runs both solve: median path time / T* of the stretched field and of the field trained on gap k, and how many runs."""
    e = runs.get((base, k, 'plus', lab, S0))
    o = own_rows(R, base, k, lab, idx)
    if not e or o is None:
        return None
    z = np.load(e['rows'])
    a, ta = z[f'{planner}_success'], z[f'{planner}_time_ratio']
    if a.shape != o[0].shape:
        return None
    both = a & o[0]
    return dict(n=int(both.sum()), stretched=float(np.median(ta[both])), own=float(np.median(o[1][both])), p90_stretched=float(np.quantile(ta[both], 0.9)), p90_own=float(np.quantile(o[1][both], 0.9)))


def seeds_of(runs, base, k, kind, lab):
    return sorted(s for (b, kk, kd, l, s) in runs if (b, kk, kd, l) == (base, k, kind, lab))


def table_data(R, runs):
    """Per base, gap, label: success of the stretched field (x, z planners; seed S0 and the training seeds' range), and the own field's."""
    T = {}
    idx = zq_index()
    for base in BASES:
        for k in KS:
            for lab in ('id', 'pmzw'):
                e = runs.get((base, k, 'plus', lab, S0))
                row = dict(own_x=pct(own(R, base, k, lab, 'x')), own_z=pct(own(R, base, k, lab, 'z')))
                pl = 'x' if lab == 'id' else 'z'
                row['own_cat'] = {c: pct(own_cat(R, base, k, lab, pl, c)) for c in CATS} if not (lab == 'id' and base == 'c22' and k == 1) else {}
                if not (lab == 'id' and base == 'c22' and k == 1):
                    row['own_x_seeds'], row['own_z_seeds'] = own_seeds(R, base, k, lab, 'x'), own_seeds(R, base, k, lab, 'z') if lab != 'id' else []
                if e:
                    for pl in ('x', 'z'):
                        v = e['plans'][pl]
                        row[pl] = pct(v['success_total'])
                        row[pl + '_cat'] = {c: pct(v['by_category'].get(c)) for c in CATS}
                        row[pl + '_wall'], row[pl + '_stall'] = v['arrived_wall'], v['not_arrived']
                        row[pl + '_time'] = v['time_ratio_median']
                        row[pl + '_seeds'] = [pct(runs[(base, k, 'plus', lab, s)]['plans'][pl]['success_total']) for s in seeds_of(runs, base, k, 'plus', lab)]
                        row[pl + '_paired'] = paired_time(R, runs, base, k, lab, pl, idx)
                if lab == 'id' and base == 'c22' and k == 1 and e:      # its training run's rows are on maze_scale's 240 queries; the identity stretch is its run on this query set
                    row['own_x'] = row.get('x')
                    row['own_x_seeds'] = row.get('x_seeds')
                    row['own_cat'] = row.get('x_cat', {})
                T[f'{base}/{k}/{lab}'] = row
            u = runs.get((base, k, 'uniform', 'id', S0))
            if u:
                T[f'{base}/{k}/uniform'] = dict(x=pct(u['plans']['x']['success_total']), x_wall=u['plans']['x']['arrived_wall'], x_stall=u['plans']['x']['not_arrived'], x_count=u['plans']['x']['by_category'])
    return T


def own_seeds(R, base, k, lab, planner):
    import zdraw10 as ZD
    out = []
    for sd in ZD.seeds(R, base, k, lab, planner):
        c = ZD.test_count(R, base, k, lab, planner, seed=sd)
        if c:
            out.append(pct(c))
    return out


def geometry():
    """Mean over the four goals of report/stretchview: {base: {k: {layer: dict(rel, rank, q, band)}}}."""
    out = {}
    for base in BASES:
        for k in KS:
            f = OUT / 'stretchview' / f'{base}_k{k}.json'
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            g = d['goals']
            out.setdefault(base, {})[k] = {key: dict(rel=100 * np.mean([x['stats'][key]['all']['rel'] for x in g]), rank=np.mean([x['stats'][key]['all']['rank'] for x in g]),
                                                     q=np.mean([x['stats'][key]['quadrants']['angle'] for x in g]), band=np.mean([x['stats'][key]['band']['angle'] for x in g]))
                                           for key in ('truth1', 'ntrl1', 'warp1', 'ntrlk', 'warpk')}
    return out


LABEL = {'id': 'ntrl-demo trained on each gap', 'ids': 'ntrl-demo ×1, stretched', 'pmzw': 'warped-maze field trained on each gap', 'pmzws': 'warped-maze field ×1, stretched', 'uni': 'ntrl-demo ×1, not stretched'}
CSS = {'id': '--s-ntrl', 'ids': '--s-ntrl', 'pmzw': '--s-pm', 'pmzws': '--s-pm', 'uni': '--s-id'}
DASH = {'ids': '6 4', 'pmzws': '6 4'}


def chart(T):
    """Line charts of success against the gap: one panel per base and planner (the page's and the gap page's chart format)."""
    panels = []
    trip = lambda v, n=900: None if v is None else [v, round(v * n / 100), n]
    for base in BASES:
        for pl, sub in (('x', 'MPPI on each maze, field looked up'), ('z', "MPPI in the field's own coordinates")):
            ser = dict(id=[], ids=[], pmzw=[], pmzws=[])
            if pl == 'x':
                ser['uni'] = []
            for k in KS:
                a, b, u = T[f'{base}/{k}/id'], T[f'{base}/{k}/pmzw'], T.get(f'{base}/{k}/uniform')
                ser['id'].append(trip(a['own_x']))
                ser['ids'].append(trip(a.get(pl)))
                ser['pmzw'].append(trip(b['own_' + pl]))
                ser['pmzws'].append(trip(b.get(pl)))
                if pl == 'x':
                    ser['uni'].append(trip(u['x']) if u else None)
            panels.append(dict(title=BNAME[base], sub=sub, series=ser))
    return dict(ks=KS, panels=panels, fields=list(LABEL), label=LABEL, css=CSS, dash=DASH, xlabel='gap between the quadrants (× 6 cells)')


def texts(T):
    """LEDE, TEXT (what was run, what came out), TEXT2 (why, limits), TABLE, VERDICT, from the numbers. Differences from rounded values."""
    G = geometry()
    f1 = lambda v: '&ndash;' if v is None else f'{v:.1f}'
    f0 = lambda v: '&ndash;' if v is None else f'{v:.0f}'
    g = lambda base, k, lab: T[f'{base}/{k}/{lab}']
    rng = lambda a: '' if not a or len(a) < 2 else f'{min(a):.1f}&ndash;{max(a):.1f}'
    kb = 12
    ks = [k for k in KS if k > 1]
    lo = lambda base, lab, pl: min(g(base, k, lab)[pl] for k in KS)
    hi = lambda base, lab, pl: max(g(base, k, lab)[pl] for k in KS)
    dq = lambda base, key, what, k=kb: G[base][k][key][what]
    cat = lambda base, k, lab, pl, c: g(base, k, lab)[pl + '_cat'][c]
    lede = (f'<p class="lede">Left is the field trained on gap &times;1, stretched onto a wider plus. Right is that maze&rsquo;s true travel time. The stretched times are far too short '
            f'(ntrl-demo at &times;{kb}: {f0(dq("c22", "ntrl1", "rel"))} % off at the median on the 22&times;22 maze), but inside the quadrants they point the right way, and the planner follows direction, not value. '
            f'Planned on &times;{kb}, ntrl-demo&rsquo;s &times;1 field solves {f1(g("c22", kb, "id")["x"])} % and {f1(g("m10", kb, "id")["x"])} % of the queries, '
            f'against {f1(g("c22", kb, "id")["own_x"])} % and {f1(g("m10", kb, "id")["own_x"])} % for ntrl-demo trained on &times;{kb} itself.</p>')
    how = ('<p><b>The stretch.</b> A pose x on gap &times;k goes to &psi;(x) on gap &times;1: along x and along y, the quadrants and the band&rsquo;s edge cell stay where they are relative to their '
           'walls, the band&rsquo;s interior (6k &minus; 2 cells) is squeezed linearly onto its 4 cells at &times;1, and the heading is kept. The field is T<sub>k</sub>(x<sub>0</sub>, x<sub>1</sub>) = '
           'T<sub>1</sub>(&psi;(x<sub>0</sub>), &psi;(x<sub>1</sub>)), with no retraining and its time values unchanged. It is the inverse of the map that carried the 22&times;22 pairs onto the wider '
           'gaps earlier. Every wall lies in the part that keeps its shape or on the outer border, so free poses map to free poses and a path on one maze is a path on the other.</p>')
    ran = ('<p><b>How it was planned.</b> The fields trained on &times;1 (ntrl-demo, and the field trained on the warped maze with warped-maze pairs) on each gap&rsquo;s own 300 queries, planning '
           'seeds 0/1/2, on both bases. &ldquo;MPPI on each maze&rdquo; steps at that maze&rsquo;s normal size, the field looked up at &psi;(x). &ldquo;MPPI in the field&rsquo;s own coordinates&rdquo; '
           'keeps &times;1&rsquo;s constant step (for the warped-maze field, in its warped coordinates), so a step over the band&rsquo;s interior is longer on the wide maze, and the path is mapped back and '
           'scored on the wide maze. The chart is training seed 87358. Controls: the same two trainings done on each gap, and the &times;1 checkpoint plugged in without the plus-shaped stretch '
           '(same normalized coordinates, so the walls miss). At &times;1 the stretch is the identity and every run matches the original run for run.</p>')
    r1 = (f'<p><b>Success almost stops falling with the gap.</b> ntrl-demo&rsquo;s &times;1 field stays at {f1(lo("c22", "id", "x"))}&ndash;{f1(hi("c22", "id", "x"))} % on the 22&times;22 maze and '
          f'{f1(lo("m10", "id", "x"))}&ndash;{f1(hi("m10", "id", "x"))} % on the 10&times;10 maze from &times;1 to &times;{kb}. Trained on each gap, ntrl-demo falls to {f1(g("c22", kb, "id")["own_x"])} % and '
          f'{f1(g("m10", kb, "id")["own_x"])} %. The gain is in the queries that end in a quadrant: at &times;{kb} diagonal queries go from {f0(ownc(T, "c22", kb, "id"))} to {f0(cat("c22", kb, "id", "x", "diagonal quadrants"))} % and '
          f'{f0(ownc(T, "m10", kb, "id"))} to {f0(cat("m10", kb, "id", "x", "diagonal quadrants"))} %, same-quadrant from {f0(ownc(T, "c22", kb, "id", "same quadrant"))} to {f0(cat("c22", kb, "id", "x", "same quadrant"))} % and '
          f'{f0(ownc(T, "m10", kb, "id", "same quadrant"))} to {f0(cat("m10", kb, "id", "x", "same quadrant"))} %. '
          f'The warped-maze field does the same in its own coordinates: {f1(lo("c22", "pmzw", "z"))}&ndash;{f1(hi("c22", "pmzw", "z"))} % and {f1(lo("m10", "pmzw", "z"))}&ndash;{f1(hi("m10", "pmzw", "z"))} %. '
          f'At &times;{kb} that is {f1(g("m10", kb, "pmzw")["z"])} % against {f1(g("m10", kb, "pmzw")["own_z"])} % for training on &times;{kb} on the 10&times;10 maze, and level on the 22&times;22 maze ({f1(g("c22", kb, "pmzw")["z"])} against {f1(g("c22", kb, "pmzw")["own_z"])} %).</p>')
    sr = lambda base, lab, key: rng(g(base, kb, lab).get(key) or [])
    r1b = (f'<p><b>Over the three training seeds at &times;{kb}:</b> ntrl-demo stretched {sr("c22", "id", "x_seeds")} % and {sr("m10", "id", "x_seeds")} %, trained there {sr("c22", "id", "own_x_seeds")} % and {sr("m10", "id", "own_x_seeds")} %; '
           f'the warped-maze field stretched {sr("c22", "pmzw", "z_seeds")} % and {sr("m10", "pmzw", "z_seeds")} %, trained there {sr("c22", "pmzw", "own_z_seeds")} % and {sr("m10", "pmzw", "own_z_seeds")} %. At &times;1 the seeds alone span '
           f'{rng(g("c22", 1, "id").get("x_seeds") or [])} % and {rng(g("m10", 1, "id").get("x_seeds") or [])} % for ntrl-demo.</p>')
    un = [T[f'{b}/{k}/uniform']['x'] for b in BASES for k in ks]
    us = [T[f'{b}/{k}/uniform']['x_count'] for b in BASES for k in ks]
    cross = sum(u[c][0] for u in us for c in ('adjacent quadrants', 'diagonal quadrants'))
    crossn = sum(u[c][1] for u in us for c in ('adjacent quadrants', 'diagonal quadrants'))
    r2 = (f'<p><b>The plus-shaped stretch is what does it.</b> The same checkpoint without the stretch solves {f1(min(un))}&ndash;{f1(max(un))} %, nearly all of it open&ndash;open queries, and {cross} of {crossn:,} runs between two quadrants.</p>')
    w1 = (f'<p><b>Why a field with the wrong times plans better.</b> MPPI compares the times of nearby poses, so it needs direction, not the value. With the true &times;1 time stretched, the descent '
          f'direction inside the quadrants is off by {f0(dq("c22", "truth1", "q"))}&deg; against the true &times;{kb} time (median, mean of four goals): the stretch costs nothing there, although the value is {f0(dq("c22", "truth1", "rel"))} % off. '
          f'ntrl-demo&rsquo;s &times;1 field is off by {f0(dq("c22", "ntrl1", "q"))}&deg; and {f0(dq("m10", "ntrl1", "q"))}&deg; there (its fitting error at &times;1), ntrl-demo trained on &times;{kb} by {f0(dq("c22", "ntrlk", "q"))}&deg; and {f0(dq("m10", "ntrlk", "q"))}&deg;. '
          f'So training on the wide maze gets the times right ({f0(dq("c22", "ntrlk", "rel"))} % off) and the quadrants wrong, which fits the earlier finding that the quadrants take only 18 % and 12 % of each axis at &times;{kb}. '
          f'In the band the trade reverses: the stretched field is off by {f0(dq("c22", "ntrl1", "band"))}&deg; and the &times;{kb} field by {f0(dq("c22", "ntrlk", "band"))}&deg;, and the stretch alone accounts for that (the true time stretched is off by {f0(dq("c22", "truth1", "band"))}&deg;), '
          f'because it squeezes the band&rsquo;s length but not the quadrants&rsquo;, so the best way across changes. Open&ndash;open queries still solve at about 100 %.</p>')
    px, pz = g('c22', kb, 'id')['x_paired'], g('c22', kb, 'id')['z_paired']
    mx = g('m10', kb, 'id')['x_paired']
    w2 = (f'<p><b>The cost shows in the paths.</b> On the runs both fields solve at &times;{kb}, ntrl-demo&rsquo;s stretched field takes {f1(100 * (px["stretched"] - 1))} % and {f1(100 * (mx["stretched"] - 1))} % longer than the optimum at the median, '
          f'and {f0(100 * (px["p90_stretched"] - 1))} % and {f0(100 * (mx["p90_stretched"] - 1))} % at the 90th percentile; the field trained on &times;{kb} is at the optimum ({px["own"]:.2f} and {mx["own"]:.2f} times it) at the median. '
          f'Those shared runs are mostly open-band queries, the only ones that field solves.</p>')
    a, b = g('c22', kb, 'pmzw'), g('m10', kb, 'pmzw')
    w3 = (f'<p><b>The one failure: the warped-maze field with the maze planner.</b> From &times;8 it stalls: {f1(a["x"])} and {f1(b["x"])} % at &times;{kb}, {a["x_stall"]} and {b["x_stall"]} of 900 runs never arrive, while the same field in its own coordinates solves '
          f'{f1(a["z"])} and {f1(b["z"])} %. Same-quadrant queries, which never cross the band, are unaffected ({f0(cat("c22", kb, "pmzw", "x", "same quadrant"))} and {f0(cat("m10", kb, "pmzw", "x", "same quadrant"))} %). '
          f'The band is squeezed 17.5 times at &times;{kb} and this field also counts a band cell as about 1/m of a quadrant cell, so a normal step changes its time too little for the softmax to prefer any direction. '
          f'That reading is arithmetic plus this one comparison: the other planner&rsquo;s longer step over the band is the main change, and I have not measured the softmax weights. '
          f'ntrl-demo&rsquo;s field has only the first factor and does not stall.</p>')
    md = lambda base, k: float(np.mean(np.array(g(base, k, 'id')['x_seeds']) - np.array(g(base, k, 'id')['z_seeds'])))
    sg = lambda v: f'{v:+.1f}'.replace('-', '&minus;')
    w4 = (f'<p><b>What this does and does not show.</b> With the constant-step planner this is the &times;1 planning problem with its path redrawn, so the flat line is close to true by construction: the &times;1 rate on another draw of queries. '
          f'The test shows that nothing about the wide gap has to be learned to solve these queries, so what training on the wide maze loses is the quadrants&rsquo; fit. It is not a method for a new maze: &psi; uses the plus layout, the same knowledge as the constructed warp, and it works here because the wide maze is the &times;1 maze with an empty band inserted. '
          f'For ntrl-demo the maze planner against the constant step, mean over the three training seeds at &times;8 and &times;{kb}: {sg(md("c22", 8))} and {sg(md("c22", kb))} points on the 22&times;22 maze, {sg(md("m10", 8))} and {sg(md("m10", kb))} on the 10&times;10 maze. '
          f'On the 22&times;22 maze that is the same direction as test 3; I have not traced it.</p>')
    head = ('<tr><th>maze</th><th>gap</th><th>ntrl-demo on this gap</th><th>ntrl-demo &times;1, MPPI on the maze</th><th>ntrl-demo &times;1, constant step</th><th>&times;1, not stretched</th>'
            '<th>warped field on this gap, own coordinates</th><th>warped field &times;1, own coordinates</th><th>warped field &times;1, MPPI on the maze</th></tr>')
    cell = lambda v, seeds=None: f1(v) + (f' <small>({rng(seeds)})</small>' if seeds and len(seeds) > 1 else '')
    rows = []
    for base in BASES:
        for k in KS:
            a, b, u = g(base, k, 'id'), g(base, k, 'pmzw'), T.get(f'{base}/{k}/uniform') or {}
            rows.append(f'<tr><td>{BNAME[base]}</td><td>&times;{k}</td><td>{cell(a["own_x"], a.get("own_x_seeds"))}</td><td>{cell(a.get("x"), a.get("x_seeds"))}</td><td>{cell(a.get("z"), a.get("z_seeds"))}</td><td>{f1(u.get("x"))}</td>'
                        f'<td>{cell(b["own_z"], b.get("own_z_seeds"))}</td><td>{cell(b.get("z"), b.get("z_seeds"))}</td><td>{cell(b.get("x"), b.get("x_seeds"))}</td></tr>')
    table = f'<p class="note">% of 900 runs, training seed 87358; in brackets the range over the training seeds where several exist.</p><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'
    verdict = (f'<li>The field trained on &times;1, stretched onto the wider gap with no retraining, keeps its &times;1 success at every gap: ntrl-demo {f1(g("c22", kb, "id")["x"])} / {f1(g("m10", kb, "id")["x"])} % at &times;{kb} against '
               f'{f1(g("c22", kb, "id")["own_x"])} / {f1(g("m10", kb, "id")["own_x"])} % trained there, the warped-maze field {f1(g("c22", kb, "pmzw")["z"])} / {f1(g("m10", kb, "pmzw")["z"])} % against {f1(g("c22", kb, "pmzw")["own_z"])} / {f1(g("m10", kb, "pmzw")["own_z"])} %. '
               f'Its times are far off and its directions inside the quadrants are not.</li>')
    return dict(LEDE=lede, HOW=how, TEXT=ran + r1 + r1b + r2, TEXT2=w1 + w2 + w3 + w4, TABLE=table, VERDICT=verdict)


def dif(a, b):
    return round(a, 1) - round(b, 1)


def ownc(T, base, k, lab, c='diagonal quadrants'):
    return T[f'{base}/{k}/{lab}'].get('own_cat', {}).get(c)


def page(T):
    t = (HERE / 'stretch_template.html').read_text()
    x = texts(T)
    i = t.index('<script>\nconst R')
    head, script = t[:i], t[i:]
    script = script.replace('/*RESULTS*/null', json.dumps(chart(T), ensure_ascii=True))
    script = ''.join(c if ord(c) < 128 else f'\\u{ord(c):04x}' for c in script)         # the local check page and the artifact both then read the same bytes
    head = head.replace('<!--LEDE-->', x['LEDE']).replace('<!--TEXT-->', x['TEXT']).replace('<!--TEXT2-->', x['TEXT2']).replace('<!--TABLE-->', x['TABLE'])
    head = head.replace('<div class="col" id="note"></div>', f'<div class="col">{x["HOW"]}<p>Dashed lines mark where the stretched part of the band starts and ends. In a quadrant view both maps use a finer grid (0.13 cells); pick &ldquo;each its own&rdquo; to compare shapes, since the left values are much smaller.</p></div>')
    (OUT / 'stretchview.html').write_text(head + script)
    return x


def main():
    R, runs = collect()
    T = table_data(R, runs)
    (OUT / 'stretch.json').write_text(json.dumps(dict(table=T, runs={'/'.join(map(str, k)): v['run'] for k, v in runs.items()}), indent=1))
    x = page(T)
    (OUT / 'stretch_text.json').write_text(json.dumps(x, indent=1))
    f = lambda v: '   –' if v is None else f'{v:5.1f}'
    for base in BASES:
        print(base)
        for k in KS:
            a, b, u = T[f'{base}/{k}/id'], T[f'{base}/{k}/pmzw'], T.get(f'{base}/{k}/uniform', {})
            print(f' x{k:<2} id own {f(a["own_x"])} str x {f(a.get("x"))} z {f(a.get("z"))} seeds {a.get("x_seeds")} | pmzw own x {f(b["own_x"])} z {f(b["own_z"])} str x {f(b.get("x"))} z {f(b.get("z"))} | uniform {f(u.get("x"))}')


if __name__ == '__main__':
    main()
