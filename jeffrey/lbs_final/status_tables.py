"""Four status tables (user request 2026-11-24): maze 2-D, gap maze 2-D, Gibson 3-D (ten houses pooled into one column), 6-DOF arm.

Columns = tests, rows = methods, sub-rows = generation time, path length (mean ± std over solved queries) and success rate.
Read from logs only, nothing is run:
  - maze / gap / Gibson: report/summary.json (report.py, the registered lbs_final bench runs).
  - 6-DOF arm: the per-query score files of the official cabinet evaluator (experiments/ntrl_cabinet/evaluate.py through
    jeffrey/lbs_arm/evaluate6.py), ntrl-demo MPPI, planning seeds 0/1/2 x 600 frozen queries, training seed 87358.
    These evaluations plan all queries as one GPU batch and log no per-query time; the time row says so and gives the
    MPPI steps to the goal instead. The October baseline runs timed the release MPPI per query (raw_timing.json); that
    is a different planner, so it goes in a note, not in the table.

python3 jeffrey/lbs_final/status_tables.py   -> report/status_tables.json (page.py reads it)
"""
import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / 'results'

ARM_COLS = [('cabinet', 'Cabinet (experiments/cabinet)'), ('rw6', 'Real-world cupboard (real_world_6dof)')]
ARM_ROWS = [  # key, name, {col: (evaluation run id, what it is)}
    ('ntrl', 'ntrl-demo', {'cabinet': ('910564a4', 'lbs_arm control, field eab861c8'), 'rw6': ('799684da', 'rw_ntrl, field 9fa94e84')}),
    ('pnt', 'P-NTFields', {'cabinet': ('53c1dd59', 'P-NTFields cabinet run, scored with ntrl-demo MPPI')}),
    ('lbs', 'LBS (Fourier ×1)', {'cabinet': ('69a9fa28', 'field 793479a8'), 'rw6': ('813a077f', 'rw_lbs, field cd9646ee')}),
    ('lbsf', 'LBS (best Fourier freq.)', {'cabinet': ('d3ecaf0a', '×0.25, field 5f629752'), 'rw6': ('a35eb399', '×0.5, field 121e678e')}),
    ('rrtc', 'RRT-Connect', {}),
    ('lazyprm', 'LazyPRM', {}),
]
REL_TIMING = [('ntrl-demo (registered best, 664b5d50)', '664b5d50', 'metric_mppi'), ('P-NTFields (53c1dd59)', '53c1dd59', 'pnt_mppi')]


def stats(v):
    if not v:
        return None
    m = sum(v) / len(v)
    return dict(mean=m, std=math.sqrt(sum((x - m) ** 2 for x in v) / len(v)), n=len(v))


def run_dir(rid):
    d = sorted(glob.glob(str(RES / f'registered_*-{rid}')))
    assert len(d) == 1, rid
    return Path(d[0])


def arm_cell(rid):
    files = sorted(glob.glob(str(run_dir(rid) / 'evaluation' / '*ntrl_mppi_*_seed[0-9].score.json')))
    assert len(files) == 18, (rid, len(files))
    n = ok = 0
    cs, ee, steps = [], [], []
    for f in files:
        S = next(iter(json.load(open(f)).values()))
        for q in S['per_query'].values():
            n += 1
            if q['success']:
                ok += 1
                cs.append(q['cs_len']); ee.append(q['ee_len']); steps.append(q['n_waypoints'])
    return dict(n=n, success=ok, pct=100 * ok / n, cs=stats(cs), ee=stats(ee), steps=stats(steps), files=len(files))


def release_timing(rid, prefix):
    """Release MPPI (a different planner from the table's): seconds per query from raw_timing.json, solved queries only."""
    d = run_dir(rid) / 'evaluation'
    t, n, ok = [], 0, 0
    for f in sorted(glob.glob(str(d / f'{prefix}_*_seed[0-9].raw_timing.json'))):
        sec = {r['query_id']: r['seconds'] for r in json.load(open(f))['rows']}
        S = next(iter(json.load(open(f.replace('.raw_timing.json', '.score.json'))).values()))
        for qid, q in S['per_query'].items():
            n += 1
            if q['success']:
                ok += 1
                t.append(sec[int(qid)])
    return dict(n=n, success=ok, time=stats(t))


def pooled(cells):
    """Pool per-house cells (each: success, n, time/length mean, std with ddof 0, n solved) into one."""
    out = dict(n=sum(c['n'] for c in cells), success=sum(c['success'] for c in cells))
    out['pct'] = 100 * out['success'] / out['n']
    for k in ('time', 'length'):
        parts = [c[k] for c in cells if c[k]]
        N = sum(p['n'] for p in parts)
        m = sum(p['n'] * p['mean'] for p in parts) / N
        var = sum(p['n'] * (p['std'] ** 2 + p['mean'] ** 2) for p in parts) / N - m * m
        out[k] = dict(mean=m, std=math.sqrt(max(var, 0)), n=N)
    out['houses'] = len(cells)
    return out


def main():
    D = json.load(open(HERE / 'report/summary.json'))
    tables = []
    for t in D['tables']:
        if t['key'] == 'gibson':
            col = dict(key='gibson', label=f'{len(t["cols"])} houses pooled', cells={})
            for m in D['methods']:
                cs = [c['cells'][m] for c in t['cols'] if c['cells'][m]]
                col['cells'][m] = pooled(cs) if cs else None
            cols = [col]
        else:
            cols = [dict(key=c['key'], label=c['label'], cells={m: (None if not c['cells'][m] else
                    dict(n=c['cells'][m]['n'], success=c['cells'][m]['success'], pct=c['cells'][m]['pct'], time=c['cells'][m]['time'],
                         length=c['cells'][m]['length'], instances=c['cells'][m]['instances'], of=c['cells'][m]['of'])) for m in D['methods']})
                    for c in t['cols']]
        tables.append(dict(key=t['key'], title=t['title'], kind='plane', rows=[(m, D['names'][m]) for m in D['methods']], cols=cols))
    arm = dict(key='arm', title='6-DOF arm (UR5e)', kind='arm', rows=[(k, n) for k, n, _ in ARM_ROWS],
               cols=[dict(key=c, label=l, cells={}) for c, l in ARM_COLS])
    for k, _, runs in ARM_ROWS:
        for col in arm['cols']:
            r = runs.get(col['key'])
            col['cells'][k] = None if not r else dict(arm_cell(r[0]), run=r[0], what=r[1])
    tables.append(arm)
    rel = [dict(name=n, run=r, **release_timing(r, p)) for n, r, p in REL_TIMING]
    out = dict(tables=tables, release_timing=rel)
    (HERE / 'report/status_tables.json').write_text(json.dumps(out, indent=1))
    for t in tables:
        for c in t['cols']:
            print(t['key'], c['label'], {m: (round(x['pct'], 1) if x else None) for m, x in c['cells'].items()})
    print(rel)



# ---------------------------------------------------------------- HTML (page.py calls section())
import html as _html

OPAQ = ' fill-opacity="0.6"'
COLORS = dict(ntrl='var(--s1)', lbs='var(--s2)', lbsf='var(--s2)', pnt='var(--ink-3)', rrtc='var(--s3)', lazyprm='var(--s4)')
INVENTORY = [
    ('LBS benchmark, 2026-11-19/20 (jeffrey/lbs, artifact P2NJd3SpZZHRZ9qLmnQCoB)',
     'LBS and its three ablations (no lift, no Bellman, no segment) next to ntrl-demo; one training seed, success only (planning seed 0, validated during training). '
     'Mazes 10 / 13 / 18; gap mazes ×1 / ×2 / ×4; 2-D Gibson (SE(2) robot) Allensville, Lakeville, Klickitat with re-runs and controls; segment variants (plain vs safe traversals) on all 9 scenes; '
     '3-D Gibson (point robot) in 5 houses, LBS vs ntrl-demo.'),
    ('LBS final results, 2026-11-22/23 (jeffrey/lbs_final, this page)',
     'ntrl-demo, LBS, RRT-Connect, LazyPRM on mazes 10 / 14 / 18 / 22 and gap mazes ×1 / ×2 / ×4 / ×8 (3 instances each) and 10 3-D Gibson houses; 500 random queries per scene; '
     'success, time, path length. Sampling planners at 10 s and 30 s per query. LBS timed with the jeffrey/lbs_fast GPU planner.'),
    ('LBS inference time, 2026-11-23 (jeffrey/lbs_fast; stopped by request)',
     'Fast MPPI variants (torch.compile, CUDA graphs, native C++), learned clearance grid, bidirectional MPPI. Clean timing only on maze 10 instance 1; the chosen planner gives the LBS row of the tables here.'),
    ('LBS on the 6-DOF cabinet arm, 2026-11-20/22 (jeffrey/lbs_arm)',
     'Cabinet scene (600 frozen queries × 3 planning seeds, official evaluator): ntrl-demo control, LBS, ablations (no S 1147, no B 1134, lift only 1115, no lift 521 of 1800), '
     'Fourier frequency ×0.25–×4, lift range h 0.05–0.71, cupboard-heavy training data and segment endpoints, second training seeds; P-NTFields and older ntrl-demo runs from October. '
     'Real-world cupboard real_world_6dof: ntrl-demo, LBS, lift only, Fourier ×0.25 / ×0.5, segment-loss leaf rules and weights 1–100, second seeds; close-pair accuracy tests; stall diagnosis.'),
    ('Two arms, 12-DOF, 2026-11-21/22 (jeffrey/lbs_arm, real_world_12dof; own scorer, no official evaluator)',
     'Straight line 55/600, Euclidean field 118/1800, ntrl-demo 30–48, lift only 194 (×1) / 666–673 (×0.25–×0.5), LBS 565–612 at ×0.25–×0.5, lift + segment 628–679.'),
    ('Two-arm placement, 2026-11-23 (jeffrey/arm_placement, rented 4090s; re-created here)',
     'Sweep of base distance 0.25–1.00 m, spacing and height for the 12-DOF pair: lift field vs ntrl-demo with forward and two-ended planners, Euclid control; '
     'zone-to-zone sets and 10 demo queries. Re-creation of p_d055_s46_h60 on this machine: 712–742 of 900 (original 733).'),
]


def _e(s):
    return _html.escape(str(s), quote=True)


def _ms(s, d=2):
    if not s:
        return ''
    f = (lambda x: f'{x:.3f}') if s['mean'] < 0.1 else (lambda x: f'{x:.{d}f}')
    return f'{f(s["mean"])} ± {f(s["std"])}'


def _cell_rows(t, k):
    """Three <td> lists (time, length, success) for row k."""
    tr, lr, sr = [], [], []
    for c in t['cols']:
        x = c['cells'].get(k)
        if x is None:
            for r in (tr, lr, sr):
                r.append('<td class="pend">not run</td>')
            continue
        part = '' if x.get('instances', 1) == x.get('of', 1) else f' ({x["instances"]}/{x["of"]})'
        sr.append(f'<td>{x["pct"]:.1f} % <span class="dim">({x["success"]:,} / {x["n"]:,}){part}</span></td>')
        if t['kind'] == 'arm':
            tr.append(f'<td class="pend">not logged <span class="dim">· {_ms(x["steps"], 0)} MPPI steps</span></td>')
            lr.append(f'<td>{_ms(x["cs"])} rad <span class="dim">· EE {_ms(x["ee"])} m</span></td>')
        else:
            tr.append(f'<td>{_ms(x["time"])} s</td>')
            lr.append(f'<td>{_ms(x["length"], 1)} cells</td>')
    return tr, lr, sr


def table_html(t):
    head = '<tr><th>method</th><th>metric</th>' + ''.join(f'<th>{_e(c["label"])}</th>' for c in t['cols']) + '</tr>'
    best = {c['key']: max([x['pct'] for x in c['cells'].values() if x] or [None]) for c in t['cols']}
    rows = []
    for k, name in t['rows']:
        tr, lr, sr = _cell_rows(t, k)
        sr = [s.replace('<td>', '<td class="best">', 1) if (t['cols'][i]['cells'].get(k) and abs(t['cols'][i]['cells'][k]['pct'] - best[t['cols'][i]['key']]) < 1e-9) else s
              for i, s in enumerate(sr)]
        key = f'<i class="key" style="background:{COLORS[k]}"></i>'
        rows.append(f'<tr class="g0"><td class="l m" rowspan="3">{key}{_e(name)}</td><td class="l sub">generation time</td>{"".join(tr)}</tr>'
                    f'<tr><td class="l sub">path length</td>{"".join(lr)}</tr>'
                    f'<tr class="g1"><td class="l sub">success rate</td>{"".join(sr)}</tr>')
    return f'<div class="tbl st"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def arm_chart(t):
    """Horizontal bars, success of every arm cell that exists, grouped by scene."""
    W, left, right, bh, gap = 1000, 250, 70, 18, 6
    items = []
    for c in t['cols']:
        items.append(('h', c['label']))
        for k, name in t['rows']:
            x = c['cells'].get(k)
            items.append(('b', name, k, x))
    H = sum(26 if it[0] == 'h' else bh + gap for it in items) + 26
    pw = W - left - right
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="6-DOF arm: success rate">']
    for v in (0, 25, 50, 75, 100):
        X = left + pw * v / 100
        out.append(f'<line x1="{X:.1f}" x2="{X:.1f}" y1="4" y2="{H - 22}" stroke="var(--grid)"/><text x="{X:.1f}" y="{H - 6}" text-anchor="middle">{v} %</text>')
    yy = 4
    for it in items:
        if it[0] == 'h':
            out.append(f'<text x="0" y="{yy + 17}" style="font-weight:600;fill:var(--ink)">{_e(it[1])}</text>')
            yy += 26
            continue
        _, name, k, x = it
        out.append(f'<text x="{left - 10}" y="{yy + bh - 4}" text-anchor="end">{_e(name)}</text>')
        if x is None:
            out.append(f'<text x="{left + 6}" y="{yy + bh - 4}" style="fill:var(--ink-3)">not run</text>')
        else:
            w = pw * x['pct'] / 100
            tip = f'<b>{_e(name)}</b><br>{x["pct"]:.1f} % ({x["success"]} of {x["n"]} runs)<br>evaluation {x["run"]} ({_e(x["what"])})'
            out.append(f'<g class="mk" tabindex="0" data-tip="{_e(tip)}"><rect class="vis" x="{left}" y="{yy}" width="{w:.1f}" height="{bh}" rx="3" fill="{COLORS[k]}"'
                       f'{OPAQ if k == "lbs" else ""}/>'
                       f'<text x="{left + w + 6:.1f}" y="{yy + bh - 4}">{x["pct"]:.1f} %</text></g>')
        yy += bh + gap
    out.append('</svg>')
    return ''.join(out)


STYLE = ('<style>.st table td.m{font-weight:600;vertical-align:top;border-bottom:1px solid var(--ink-3)} .st td.sub{color:var(--ink-2)} '
         '.st tr.g1 td{border-bottom:1px solid var(--ink-3)} .st .dim{color:var(--ink-3);font-size:0.92em} .st table{font-size:13.5px} '
         'details.open-tbl > summary{font-size:16px}</style>')


def section():
    f = HERE / 'report/status_tables.json'
    if not f.exists():
        return ''
    S = json.load(open(f))
    T = {t['key']: t for t in S['tables']}
    rel = '; '.join(f'{r["name"]} {r["success"]}/{r["n"]} solved in {r["time"]["mean"]:.3f} ± {r["time"]["std"]:.3f} s' for r in S['release_timing'])
    notes = {
        'maze': 'Each column pools 3 mazes × 500 random queries (1,500). Time and length: mean ± standard deviation over the solved queries.',
        'gap': 'Each column pools 3 base mazes × 500 queries (1,500); half of the goals inside a maze block, half in the open band.',
        'gibson': f'The {T["gibson"]["cols"][0]["cells"]["lbs"]["houses"]} 3-D houses of the tables below pooled into one column (500 queries each, 5,000 in all); '
                  'mean and standard deviation over every solved query of every house (per-house numbers in the Gibson section below).',
        'arm': ('Official cabinet evaluator (experiments/ntrl_cabinet/evaluate.py), 600 frozen certified queries × planning seeds 0/1/2 = 1,800 runs per cell, '
                'ntrl-demo’s MPPI on every field, training seed 87358. <b>Generation time was not logged</b>: these evaluations plan all queries as one GPU batch '
                'and record no per-query time; the cell gives the MPPI steps of the solved paths instead (same planner and step size for the ntrl-demo and LBS rows; P-NTFields ran in October with its own normalization). '
                f'The only per-query times on this suite are from the October runs with the release MPPI, a different planner: {rel}. '
                'Path length = joint-space length in radians; EE = end-effector path in metres. P-NTFields exists for the cabinet only. '
                'RRT-Connect and LazyPRM have not been benchmarked on the arm (the suites were certified with RRT-Connect, but that run was not timed as a planner). '
                'Straight line (control): cabinet 202 / 600, real-world 226 / 600. Other LBS variants on these scenes (ablations, seeds, knobs) are in jeffrey/lbs_arm/README.md.'),
    }
    h = [STYLE, '<h2 id="status">Where things stand: four tables</h2>',
         '<p class="lede">Every cell is read from registered runs and their logs; nothing new was run for this section. '
         'Generation time is wall time per solved query: LBS on the RTX 5070 with the fast GPU planner, ntrl-demo, RRT-Connect and LazyPRM on two CPU threads, '
         'the sampling planners with 30 s per query. Path length in cells on the 2-D and 3-D scenes (one cell = 0.25 m in the houses).</p>']
    for key, title in (('maze', 'Table 1 · Maze, 2-D'), ('gap', 'Table 2 · Gap maze, 2-D (10 × 10 base)'), ('gibson', 'Table 3 · Gibson houses, 3-D'), ('arm', 'Table 4 · 6-DOF arm (UR5e + 2F-140)')):
        t = T[key]
        h.append(f'<details class="open-tbl" open><summary>{title}</summary>')
        if key == 'arm':
            h.append(f'<div class="tbl"><div class="chart" style="margin:4px 0 10px"><p class="ct">Success rate, 6-DOF arm</p>'
                     f'<p class="cs">Share of 1,800 planning runs solved; “not run” rows have no data yet.</p>{arm_chart(t)}</div></div>')
        h.append(table_html(t))
        h.append(f'<div class="tbl"><p class="note">{notes[key]}</p></div></details>')
    inv = ''.join(f'<li><b>{_e(a)}</b>: {_e(b)}</li>' for a, b in INVENTORY)
    h.append(f'<details><summary>Every LBS-related test run so far</summary><div class="tbl"><ul>{inv}</ul></div></details>')
    return '\n'.join(h)


if __name__ == '__main__':
    main()
