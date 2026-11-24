"""report/summary.json (report.py) -> report/lbs_fast.html, the artifact "2026-11-23 LBS Inference Time"."""
import html
import json
import math
from pathlib import Path
from page_style import CSS, JS

HERE = Path(__file__).resolve().parent
TITLE = '2026-11-23 LBS Inference Time'
FAM = ['maze10', 'maze14', 'maze18', 'maze22', 'gap1', 'gap2', 'gap4', 'gap8', 'houses']
LADDER = ['maze10_1', 'maze22_1', 'gap4_1', 'Onaga3d']
SCENE_LABEL = dict(maze10_1='maze 10 × 10', maze22_1='maze 22 × 22', gap4_1='gap maze ×4', Onaga3d='house Onaga (3-D)')


def esc(s):
    return html.escape(str(s), quote=True)


def ms(x):
    """Milliseconds from seconds, compact."""
    v = 1e3 * x
    return f'{v:.0f}' if v >= 100 else f'{v:.1f}' if v >= 10 else f'{v:.2f}'


def num(v):
    return f'{v:.0f}' if v >= 100 else f'{v:.1f}' if v >= 10 else f'{v:.2f}' if v >= 1 else f'{v:.3f}'


def legend(series):
    return '<div class="legend">' + ''.join(f'<span><i class="key" style="background:{c}"></i>{esc(n)}</span>' for n, c in series) + '</div>'


def nice_ticks(top, n=5):
    if top <= 0:
        return [0, 1]
    step = 10 ** math.floor(math.log10(top / n))
    step = next(s * step for s in (1, 2, 2.5, 5, 10) if top / (s * step) <= n)
    return [i * step for i in range(int(math.ceil(top / step - 1e-9)) + 1)]


def hbars(items, unit, aria, base=None, color='var(--s1)'):
    """Horizontal bars, one series, linear from zero. items: (label, value, tip, highlight). Direct labels: value and factor to base."""
    W, row, left, right, top = 1000, 30, 330, 150, 6
    H = top + row * len(items) + 26
    vmax = max(v for _, v, _, _ in items)
    ticks = nice_ticks(vmax)
    hi = ticks[-1]
    pw = W - left - right
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}">']
    for t in ticks:
        x = left + pw * t / hi
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{H - 24}" stroke="var(--grid)" stroke-width="1"/><text x="{x:.1f}" y="{H - 8}" text-anchor="middle">{t:g}</text>')
    out.append(f'<text x="{W - right + 8}" y="{H - 8}">{esc(unit)}</text>')
    for i, (label, v, tip, hl) in enumerate(items):
        y = top + i * row + 5
        h = row - 10
        w = max(pw * v / hi, 1.5)
        r = min(4, w / 2)
        d = f'M{left},{y} H{left + w - r:.1f} Q{left + w:.1f},{y} {left + w:.1f},{y + r} V{y + h - r} Q{left + w:.1f},{y + h} {left + w - r:.1f},{y + h} H{left} Z'
        fac = f' · {base / v:.1f}×' if base and v and abs(base / v - 1) > 0.02 else ''
        out.append(f'<g class="mk" tabindex="0" data-tip="{esc(tip)}"><rect class="hit" x="0" y="{y - 5}" width="{W}" height="{row}"/>'
                   f'<text x="{left - 10}" y="{y + h / 2 + 4:.1f}" text-anchor="end" style="fill:var(--ink){";font-weight:600" if hl else ""}">{esc(label)}</text>'
                   f'<path class="vis" d="{d}" fill="{color}"/>'
                   f'<text x="{left + w + 8:.1f}" y="{y + h / 2 + 4:.1f}" style="fill:var(--ink){";font-weight:600" if hl else ""}">{num(v)}{fac}</text></g>')
    out.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{H - 24}" stroke="var(--ink-3)" stroke-width="1"/></svg>')
    return ''.join(out)


def grouped(cols, series, get, kind, aria, unit=''):
    """Columns (key, label) x series (key, name, color). get(col, series) -> None or dict(v, tip[, lo, hi]). kind 'pct' = bars from zero,
    'log' = dots on a log axis with an optional lo-hi line."""
    W, H, left, right, top, bottom = 1000, (230 if kind == 'pct' else 260), 60, 12, 12, 30
    pw, ph = W - left - right, H - top - bottom
    gw = pw / len(cols)
    bw = min(22, gw * 0.8 / len(series) - 2)
    cells = {(c[0], s[0]): get(c[0], s[0]) for c in cols for s in series}
    vals = [x for c in cells.values() if c for x in (c['v'], c.get('lo'), c.get('hi')) if x]
    if kind == 'pct':
        lo, hi, ticks = 0, 100, [0, 25, 50, 75, 100]
    else:
        pos = [v for v in vals if v > 0] or [1, 10]
        lo = 10 ** math.floor(math.log10(min(pos) * 0.9))
        hi = 10 ** math.ceil(math.log10(max(pos) * 1.05))
        ticks = [10 ** k for k in range(int(round(math.log10(lo))), int(round(math.log10(hi))) + 1)]

    def y(v):
        if kind == 'log':
            v = min(max(v, lo), hi)
            return top + ph - ph * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return top + ph - ph * (v - lo) / (hi - lo)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}">']
    for v in ticks:
        lab = f'{v:g} %' if kind == 'pct' else f'{v:g}{unit}'
        out.append(f'<line x1="{left}" x2="{W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)" stroke-width="1"/><text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{lab}</text>')
    for gi, (ck, cl) in enumerate(cols):
        cx = left + gi * gw + gw / 2
        out.append(f'<text x="{cx:.1f}" y="{H - bottom + 18}" text-anchor="middle">{esc(cl)}</text>')
        for si, (sk, sn, col) in enumerate(series):
            c = cells[(ck, sk)]
            if not c:
                continue
            bx = cx + (si - (len(series) - 1) / 2) * (bw + 2)
            if kind == 'pct':
                y0, y1 = y(0), y(c['v'])
                h = max(y0 - y1, 0.5)
                r = min(4, h / 2, bw / 2)
                d = (f'M{bx - bw / 2:.1f},{y0:.1f} V{y1 + r:.1f} Q{bx - bw / 2:.1f},{y1:.1f} {bx - bw / 2 + r:.1f},{y1:.1f} H{bx + bw / 2 - r:.1f} '
                     f'Q{bx + bw / 2:.1f},{y1:.1f} {bx + bw / 2:.1f},{y1 + r:.1f} V{y0:.1f} Z')
                out.append(f'<g class="mk" tabindex="0" data-tip="{esc(c["tip"])}"><rect class="hit" x="{bx - bw / 2 - 1:.1f}" y="{top}" width="{bw + 2:.1f}" height="{ph}"/><path class="vis" d="{d}" fill="{col}"/></g>')
            else:
                line = ''
                if c.get('lo') and c.get('hi'):
                    line = f'<line x1="{bx:.1f}" x2="{bx:.1f}" y1="{y(c["lo"]):.1f}" y2="{y(c["hi"]):.1f}" stroke="{col}" stroke-width="2" stroke-linecap="round"/>'
                out.append(f'<g class="mk" tabindex="0" data-tip="{esc(c["tip"])}"><rect class="hit" x="{bx - bw / 2 - 1:.1f}" y="{top}" width="{bw + 2:.1f}" height="{ph}"/>{line}'
                           f'<circle class="vis" cx="{bx:.1f}" cy="{y(c["v"]):.1f}" r="5" fill="{col}" stroke="var(--surface)" stroke-width="2"/></g>')
    out.append(f'<line x1="{left}" x2="{W - right}" y1="{top + ph:.1f}" y2="{top + ph:.1f}" stroke="var(--ink-3)" stroke-width="1"/></svg>')
    return ''.join(out)


def lines(xs, series, aria, xlabel, unit, log=False, W=480, H=250, ymax=None, pct=False):
    """series: (name, color, [values or None], [tips]). xs: labels (categorical, evenly spaced)."""
    left, right, top, bottom = 52, 96, 12, 40
    pw, ph = W - left - right, H - top - bottom
    vals = [v for _, _, vs, _ in series for v in vs if v is not None]
    if pct:
        lo, hi, ticks = 0, 100, [0, 25, 50, 75, 100]
    elif log:
        lo = 10 ** math.floor(math.log10(min(vals) * 0.9))
        hi = 10 ** math.ceil(math.log10(max(vals) * 1.05))
        ticks = [10 ** k for k in range(int(round(math.log10(lo))), int(round(math.log10(hi))) + 1)]
    else:
        ticks = nice_ticks(ymax or max(vals), 4)
        lo, hi = 0, ticks[-1]

    def y(v):
        if log:
            return top + ph - ph * (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return top + ph - ph * (v - lo) / (hi - lo)

    def x(i):
        return left + pw * (i + 0.5) / len(xs)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}">']
    for t in ticks:
        out.append(f'<line x1="{left}" x2="{W - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="var(--grid)" stroke-width="1"/><text x="{left - 7}" y="{y(t) + 4:.1f}" text-anchor="end">{t:g}{" %" if pct else unit}</text>')
    for i, l in enumerate(xs):
        out.append(f'<text x="{x(i):.1f}" y="{H - bottom + 17}" text-anchor="middle">{esc(l)}</text>')
    out.append(f'<text x="{left + pw / 2:.1f}" y="{H - 6}" text-anchor="middle" style="fill:var(--ink-3)">{esc(xlabel)}</text>')
    ends = []
    for name, col, vs, tips in series:
        pts = [(x(i), y(v)) for i, v in enumerate(vs) if v is not None]
        if len(pts) > 1:
            out.append('<path d="M' + ' L'.join(f'{a:.1f},{b:.1f}' for a, b in pts) + f'" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for i, v in enumerate(vs):
            if v is None:
                continue
            out.append(f'<g class="mk" tabindex="0" data-tip="{esc(tips[i])}"><circle class="hit" cx="{x(i):.1f}" cy="{y(v):.1f}" r="12"/>'
                       f'<circle class="vis" cx="{x(i):.1f}" cy="{y(v):.1f}" r="4.5" fill="{col}" stroke="var(--surface)" stroke-width="2"/></g>')
        if pts:
            ends.append([pts[-1][1], name])
    ends.sort()
    for i in range(1, len(ends)):                                  # keep the end labels apart
        if ends[i][0] - ends[i - 1][0] < 14:
            ends[i][0] = ends[i - 1][0] + 14
    for yy, name in ends:
        out.append(f'<text x="{W - right + 8}" y="{yy + 4:.1f}" style="fill:var(--ink)">{esc(name)}</text>')
    out.append(f'<line x1="{left}" x2="{W - right}" y1="{top + ph:.1f}" y2="{top + ph:.1f}" stroke="var(--ink-3)" stroke-width="1"/></svg>')
    return ''.join(out)


def table(head, rows):
    return ('<div class="tbl"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td class="l">{c}</td>' if i == 0 or (isinstance(c, str) and c.startswith('<code')) else f'<td>{c}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
            + '</tbody></table></div>')


def chart_box(title, sub, svg):
    return f'<div class="chart"><p class="ct">{title}</p><p class="cs">{sub}</p>{svg}</div>'


def v_of(D, sec, scene, tag, label):
    return D.get(sec, {}).get(scene, {}).get(tag, {}).get('variants', {}).get(label)


def fam_cell(D, fam, key):
    return D['families'].get(fam, {}).get('cells', {}).get(key)


# ------------------------------------------------------------------------------------------------------------------ sections
HEAD_SERIES = [('ref:clean', 'Reference planner, CPU 2 threads', 'var(--s1)'), ('cpu:nat8', 'Fast, CPU 8 threads', 'var(--s2)'),
               ('cpu:nat8_meet_m2_s25', 'Fast + bidirectional, 2× step, 25 samples, CPU', 'var(--s5)'),
               ('gpu:goal', 'Fast, GPU', 'var(--s3)'), ('gpu:meet_m2', 'Fast + bidirectional, 2× step, GPU', 'var(--s4)')]


def fam_cols(D):
    return [(k, D['families'][k]['title'].replace('3-D houses (10)', 'houses')) for k in FAM if k in D['families'] and D['families'][k]['cells']]


def headline(D):
    cols = fam_cols(D)
    if not cols:
        return '<p class="note">The all-scene runs have not finished yet.</p>'

    def gt(c, s):
        x = fam_cell(D, c, s)
        if not x or 't_ok' not in x:
            return None
        t = x['t_ok']
        return dict(v=1e3 * t['mean'], lo=1e3 * t['median'], hi=1e3 * t['p90'],
                    tip=f'<b>{esc(dict((a, b) for a, b, _ in HEAD_SERIES)[s])}</b> · {esc(D["families"][c]["title"])}<br>mean {ms(t["mean"])} ms per solved query (median {ms(t["median"])}, 90th percentile {ms(t["p90"])})'
                        f'<br>{x["ms_per_step"]:.3f} ms per step, {x["steps_ok"]:.0f} steps')

    def gs(c, s):
        x = fam_cell(D, c, s)
        if not x:
            return None
        return dict(v=x['pct'], tip=f'<b>{esc(dict((a, b) for a, b, _ in HEAD_SERIES)[s])}</b> · {esc(D["families"][c]["title"])}<br>{x["pct"]:.1f} % solved ({x["success"]} of {x["n"]})')
    h = [legend([(n, c) for _, n, c in HEAD_SERIES]),
         '<div class="charts wide">' + chart_box('Time to generate one path', 'Milliseconds per solved query, mean (dot) with the median-to-90th-percentile range (line), log scale. Scoring is off the clock.',
                                                  grouped(cols, HEAD_SERIES, gt, 'log', 'time per solved query by scene family', ' ms'))
         + chart_box('Control: queries solved', 'Share of the same queries whose generated path reaches the goal collision-free (exact scorer, run after timing).',
                     grouped(cols, HEAD_SERIES, gs, 'pct', 'queries solved by scene family')) + '</div>']
    keys = [('ref:recorded', 'Reference as recorded in LBS Final Results (shared machine)'), ('ref:clean', 'Reference, idle machine'), ('cpu:nat8', 'Fast, CPU native 8 threads'),
            ('cpu:nat8_meet_m2', 'Fast bidirectional 2× step, CPU native'), ('cpu:nat8_m2_s25', 'Fast one-way 2× step, 25 samples, CPU native'),
            ('cpu:nat8_meet_m2_s25', 'Fast bidirectional 2× step, 25 samples, CPU native'), ('gpu:goal', 'Fast, GPU'),
            ('gpu:meet', 'Fast bidirectional, GPU'), ('gpu:meet_m2', 'Fast bidirectional 2× step, GPU'), ('gpu:goal_grid', 'Fast, GPU, learned clearance'),
            ('gpu:meet_m2_grid', 'Fast bidirectional 2× step, GPU, learned clearance')]
    rows = []
    for c, cl in cols:
        base = fam_cell(D, c, 'ref:clean')
        for k, name in keys:
            x = fam_cell(D, c, k)
            if not x or 't_ok' not in x:
                continue
            sp = f'{base["t_ok"]["mean"] / x["t_ok"]["mean"]:.1f}×' if base and 't_ok' in base else ''
            rows.append([esc(cl), esc(name), f'{x["pct"]:.1f} % ({x["success"]}/{x["n"]})', ms(x['t_ok']['mean']), ms(x['t_ok']['median']), ms(x['t_all']['mean']), f'{x["ms_per_step"]:.3f}',
                         f'{x["steps_ok"]:.0f}', sp])
    h.append('<details><summary>Full table: every planner on every scene family</summary>'
             + table(['scene', 'planner', 'solved', 'ms per solved query (mean)', 'median', 'ms per query, all queries', 'ms per step', 'steps (solved)', 'faster than the reference'], rows)
             + '</details>')
    return '\n'.join(h)


LAD_CPU = [('cpu2', 'ref', 'Reference (LBS Final Results planner), 2 threads'), ('cpu8', 'ref', 'Reference, 8 threads'),
           ('cpu2', 'cache', '+ goal encoded once, weights frozen, no autograd'), ('cpu2', 'compiled', '+ whole step compiled (torch.compile)'),
           ('cpu2', 'nat1', 'Native C++ kernel, 1 thread'), ('cpu2', 'nat2', 'Native C++ kernel, 2 threads'), ('cpu2', 'nat4', 'Native C++ kernel, 4 threads'),
           ('cpu2', 'nat8', 'Native C++ kernel, 8 threads')]
LAD_GPU = [('gpu', 'ref', 'Reference planner moved to the GPU'), ('gpu', 'cache', '+ goal encoded once, weights frozen, no autograd'),
           ('gpu', 'compiled', '+ whole step compiled'), ('gpu', 'graph', '+ CUDA graph replay'), ('gpu', 'graph_k8', '+ 8 steps per call, one read-back'),
           ('gpu', 'graph_k8_bulk', '+ one random draw per call'), ('gpu', 'graph_k16_bulk', '16 steps per call')]


def ladder(D, scene):
    h = []
    base = v_of(D, 'ladder', scene, 'cpu2', 'ref')
    for dev, spec, title in (('CPU (Ryzen 7 7800X3D)', LAD_CPU, 'CPU'), ('GPU (RTX 3090)', LAD_GPU, 'GPU')):
        items = []
        for tag, label, name in spec:
            x = v_of(D, 'ladder', scene, tag, label)
            if not x:
                continue
            same = f'<br>stops at the reference’s step on {x["same_as_ref"][0]} of {x["same_as_ref"][1]} queries' if 'same_as_ref' in x else ''
            tip = (f'<b>{esc(name)}</b><br>{x["ms_per_step"]:.3f} ms per MPPI step · {x["success"]} of {x["n"]} solved'
                   + (f' · {ms(x["t_ok"]["mean"])} ms per solved query' if 't_ok' in x else '') + same)
            items.append((name, x['ms_per_step'], tip, label in ('nat8', 'graph_k8_bulk')))
        if items:
            h.append(chart_box(f'{title}: milliseconds per MPPI step', f'{esc(SCENE_LABEL.get(scene, scene))}, first 100 queries, one query at a time. The factor is against the 2-thread CPU reference.',
                               hbars(items, 'ms per step', f'{title} ms per step by optimization', base['ms_per_step'] if base else None)))
    return '<div class="charts wide">' + ''.join(h) + '</div>'


def ladder_table(D):
    rows = []
    for scene in LADDER:
        for tag, e in D['ladder'].get(scene, {}).items():
            for label, x in e['variants'].items():
                rows.append([esc(SCENE_LABEL.get(scene, scene)), esc(tag), esc(label), f'{x["success"]}/{x["n"]}', f'{x["ms_per_step"]:.3f}', f'{x["steps_all"]:.0f}',
                             ms(x['t_ok']['mean']) if 't_ok' in x else '', ms(x['t_all']['mean']), (f'{x["same_as_ref"][0]}/{x["same_as_ref"][1]}' if 'same_as_ref' in x else ''),
                             f'{x["setup_seconds"]:.0f}', f'<code>{e["run"]}</code>'])
    return table(['scene', 'job', 'variant', 'solved', 'ms per step', 'steps (all)', 'ms per solved query', 'ms per query (all)', 'same stop as reference', 'one-off setup (s)', 'run'], rows)


def threads_chart(D):
    out = []
    for scene in LADDER:
        ser = []
        for name, col, pick in (('reference', 'var(--s1)', lambda t: v_of(D, 'ladder', scene, f'cpu{t}', 'ref')),
                                ('compiled', 'var(--s3)', lambda t: v_of(D, 'ladder', scene, f'cpu{t}', 'compiled')),
                                ('native', 'var(--s2)', lambda t: v_of(D, 'ladder', scene, 'cpu2', f'nat{t}'))):
            vs = [pick(t) for t in (1, 2, 4, 8)]
            if any(vs):
                ser.append((name, col, [v['ms_per_step'] if v else None for v in vs], [f'<b>{name}</b>, {t} threads<br>{v["ms_per_step"]:.3f} ms per step' if v else '' for t, v in zip((1, 2, 4, 8), vs)]))
        if ser:
            out.append(chart_box(esc(SCENE_LABEL.get(scene, scene)), 'ms per MPPI step, log scale', lines(['1', '2', '4', '8'], ser, f'ms per step by threads, {scene}', 'CPU threads', '', log=True)))
    return '<div class="charts">' + ''.join(out) + '</div>'


def knob_charts(D):
    out = []
    for scene in LADDER:
        e = D['knobs'].get(scene, {}).get('gpu')
        if not e:
            continue
        V = e['variants']
        ms_ = ['1', '1.5', '2', '3']
        for what, sub, f, pct in (('solved', 'share of the first 300 queries', lambda x: x['pct'], True), ('time', 'ms per solved query, log scale', lambda x: 1e3 * x['t_ok']['mean'] if 't_ok' in x else None, False)):
            ser = []
            for name, col, pre in (('one-way', 'var(--s3)', 'goal'), ('bidirectional', 'var(--s4)', 'meet')):
                xs = [V.get(pre if m == '1' else f'{pre}_m{m}') for m in ms_]
                ser.append((name, col, [f(x) if x else None for x in xs],
                            [f'<b>{name}</b>, step × {m}<br>{x["pct"]:.1f} % solved · {ms(x["t_ok"]["mean"]) if "t_ok" in x else "-"} ms per solved query · {x["steps_ok"]:.0f} steps' if x else '' for m, x in zip(ms_, xs)]))
            out.append(chart_box(f'{esc(SCENE_LABEL.get(scene, scene))}: {what}', sub, lines(ms_, ser, f'{what} by step multiplier, {scene}', 'step multiplier', '' if pct else ' ms', log=not pct, pct=pct)))
    return '<div class="charts">' + ''.join(out) + '</div>'


def knob_table(D):
    rows = []
    for scene in LADDER:
        e = D['knobs'].get(scene, {}).get('gpu')
        if not e:
            continue
        for label, x in e['variants'].items():
            how = ', '.join(f'{k} {v}' for k, v in x['how'].items())
            rows.append([esc(SCENE_LABEL.get(scene, scene)), esc(label), f'{x["pct"]:.1f} % ({x["success"]}/{x["n"]})', f'{x["ms_per_step"]:.3f}', f'{x["steps_ok"]:.0f}' if 'steps_ok' in x else '',
                         ms(x['t_ok']['mean']) if 't_ok' in x else '', ms(x['t_all']['mean']), f'{x["len_ok"]:.2f}' if 'len_ok' in x else '', f'{x["stalls"]} / {x["collisions"]}', esc(how), f'<code>{e["run"]}</code>'])
    return table(['scene', 'variant', 'solved', 'ms per step', 'steps (solved)', 'ms per solved query', 'ms per query (all)', 'path length (cells)', 'not arrived / arrived through an obstacle',
                  'how the path ended', 'run'], rows)


def dist_section(D):
    h = []
    cols = fam_cols(D)
    ser = [('gpu:goal', 'one-way, exact clearance', 'var(--s3)'), ('gpu:goal_grid', 'one-way, learned clearance', 'var(--s1)'), ('gpu:meet_m2', 'bidirectional 2× step, exact', 'var(--s4)'),
           ('gpu:meet_m2_grid', 'bidirectional 2× step, learned', 'var(--s2)')]
    names = {a: b for a, b, _ in ser}
    if cols:
        def gs(c, s):
            x = fam_cell(D, c, s)
            return dict(v=x['pct'], tip=f'<b>{esc(names[s])}</b> · {esc(D["families"][c]["title"])}<br>{x["pct"]:.1f} % solved ({x["success"]} of {x["n"]})') if x else None

        def gt(c, s):
            x = fam_cell(D, c, s)
            return dict(v=1e3 * x['ms_per_step'], tip=f'<b>{esc(names[s])}</b> · {esc(D["families"][c]["title"])}<br>{1e3 * x["ms_per_step"]:.1f} µs per MPPI step') if x else None
        h.append(legend([(n, c) for _, n, c in ser]))
        h.append('<div class="charts wide">' + chart_box('Queries solved with the learned clearance in place of the exact one', 'GPU planners, the same queries; the learned field replaces every clearance query, including the goal’s.',
                                                          grouped(cols, ser, gs, 'pct', 'solved, exact against learned clearance'))
                 + chart_box('Step time', 'Microseconds per MPPI step on the GPU, log scale.', grouped(cols, ser, gt, 'log', 'step time, exact against learned clearance', ' µs')) + '</div>')
    rows = []
    for scene, d in D['dist'].items():
        for tag, r in d.items():
            e = r['error']
            rows.append([esc(scene), esc(tag), f'{r["parameters"] / 1e6:.2f} M', f'{r["train_seconds"]:.0f}', f'{e["mae_band"]:.5f}', f'{e["p95_band"]:.5f}', f'{e["max_band"]:.4f}',
                         f'{100 * e["free_called_blocked"]:.2f} %', f'{e["h_mae_band_cells"]:.4f}', f'<code>{r["run"]}</code>'])
    h.append('<details><summary>Learned clearance fields: error per scene</summary>'
             + table(['scene', 'model', 'parameters', 'training (s)', 'mean abs. error in the lift band (cells)', '95th percentile', 'max', 'free called blocked', 'lift height error in the band (cells)', 'run'], rows)
             + '</details>')
    rows = []
    for scene, jobs in D['distcmp'].items():
        for tag, e in jobs.items():
            for label, x in e['variants'].items():
                rows.append([esc(SCENE_LABEL.get(scene, scene)), esc(e['device']), esc(label), f'{x["pct"]:.1f} % ({x["success"]}/{x["n"]})', f'{x["ms_per_step"]:.3f}',
                             ms(x['t_ok']['mean']) if 't_ok' in x else '', f'{x["stalls"]} / {x["collisions"]}', f'<code>{e["run"]}</code>'])
    if rows:
        h.append('<details><summary>Grid against MLP as the learned clearance model (three scenes)</summary>'
                 + table(['scene', 'device', 'variant', 'solved', 'ms per step', 'ms per solved query', 'not arrived / through an obstacle', 'run'], rows) + '</details>')
    return '\n'.join(h)


def bidir_section(D):
    cols = fam_cols(D)
    if not cols:
        return ''
    ser = [('gpu:goal', 'one-way', 'var(--s3)'), ('gpu:meet', 'bidirectional', 'var(--s4)'), ('gpu:meet_m2', 'bidirectional, 2× step', 'var(--s2)')]
    names = {a: b for a, b, _ in ser}

    def gs(c, s):
        x = fam_cell(D, c, s)
        return dict(v=x['pct'], tip=f'<b>{esc(names[s])}</b> · {esc(D["families"][c]["title"])}<br>{x["pct"]:.1f} % solved ({x["success"]} of {x["n"]})') if x else None

    def gst(c, s):
        x = fam_cell(D, c, s)
        return dict(v=x['steps_ok'], tip=f'<b>{esc(names[s])}</b> · {esc(D["families"][c]["title"])}<br>{x["steps_ok"]:.0f} iterations per solved query · {x["ms_per_step"]:.3f} ms each') if x and 'steps_ok' in x else None
    return (legend([(n, c) for _, n, c in ser]) + '<div class="charts wide">'
            + chart_box('Sequential iterations per solved query', 'GPU. One iteration moves every agent once; log scale.', grouped(cols, ser, gst, 'log', 'iterations per solved query'))
            + chart_box('Queries solved', 'Share of the same queries.', grouped(cols, ser, gs, 'pct', 'solved, one-way against bidirectional')) + '</div>')


def direction_section(D):
    sc = [(k, k.replace('3d', '').replace('_', ' #').replace('maze', 'maze ').replace('gap', 'gap ×')) for k in D.get('direction', {}) if 'direction' in D['direction'][k].get('gpu', {})]
    if not sc:
        return ''
    ser = [('forward', 'one-way, start → goal', 'var(--s3)'), ('backward', 'one-way, goal → start', 'var(--s1)'), ('either', 'either one-way direction', 'var(--s5)'),
           ('meet', 'bidirectional, chasing', 'var(--s4)'), ('both', 'bidirectional, fixed targets', 'var(--s2)')]
    names = {a: b for a, b, _ in ser}

    def g(c, s):
        d = D['direction'][c]['gpu']['direction']
        if s not in d:
            return None
        return dict(v=100 * d[s] / d['n'], tip=f'<b>{esc(names[s])}</b> · {esc(c)}<br>{d[s]} of {d["n"]} queries solved')
    rows = []
    for c, cl in sc:
        d = D['direction'][c]['gpu']['direction']
        rows.append([esc(cl), d['n'], d['forward'], d['backward'], d['both_dirs'], d['either'], d['meet'], d['meet_only'], d['either_only'], d.get('both', ''), f'<code>{D["direction"][c]["gpu"]["run"]}</code>'])
    return (legend([(n, c) for _, n, c in ser]) + '<div class="charts wide">'
            + chart_box('Is the bidirectional gain "two chances"?', 'Queries solved (first 250) on scenes where the one-way planner misses some. "Either" counts a query solved if the forward or the backward one-way run solves it.',
                        grouped(sc, ser, g, 'pct', 'solved by direction')) + '</div>'
            + '<details><summary>Direction control: counts per scene</summary>'
            + table(['scene', 'queries', 'forward', 'backward', 'both directions', 'either direction', 'bidirectional (chasing)', 'chasing only', 'either only', 'bidirectional (fixed targets)', 'run'], rows) + '</details>')


def scene_table(D):
    rows = []
    for scene, jobs in D['final'].items():
        lf = jobs.get('lbs_final')
        if lf and 'clean' in lf:
            x = lf['clean']
            rows.append([esc(scene), 'reference (CPU 2 threads)', f'{x["pct"]:.1f} %', ms(x['t_ok']['mean']) if 't_ok' in x else '', f'{lf["ms_per_step_clean"]:.3f}', f'{x.get("steps_ok", 0):.0f}', f'<code>{lf["run"]}</code>'])
        for tag in ('cpu', 'gpu'):
            e = jobs.get(tag)
            if not e:
                continue
            for label, x in e['variants'].items():
                if label == 'ref':
                    continue
                rows.append([esc(scene), esc(f'{tag}: {label}'), f'{x["pct"]:.1f} %', ms(x['t_ok']['mean']) if 't_ok' in x else '', f'{x["ms_per_step"]:.3f}', f'{x.get("steps_ok", 0):.0f}', f'<code>{e["run"]}</code>'])
    return table(['scene', 'planner', 'solved', 'ms per solved query', 'ms per step', 'steps (solved)', 'run'], rows)


def body(D, T):
    n_final = sum(1 for s in D['final'].values() for t in ('gpu', 'cpu') if t in s)
    h = [f'<title>{TITLE}</title>', '<meta name="claude:viewport-fit-sides" content="cover">',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
         f'<style>{CSS}</style>', '<div class="wrap">',
         '<header class="top"><div class="eyebrow">jeffrey/lbs_fast · time to generate one path on an LBS field · RTX 3090 and Ryzen 7 7800X3D · every number from registered runs</div>',
         '<h1>LBS inference time</h1>', f'<p class="lede">{T.get("lede", "")}</p></header>']
    if T.get('status'):
        h.append(f'<div class="status">{T["status"]}</div>')
    h.append('<h2 id="answers">What the three tasks gave</h2><div class="answers">')
    for q, a in T.get('answers', []):
        h.append(f'<div><div class="q">{q}</div><div class="a">{a}</div></div>')
    h.append('</div>')
    h.append('<h2 id="all">Every scene: time per path</h2>')
    h.append(f'<p>{T.get("all", "")}</p>')
    h.append(headline(D))
    h.append('<details><summary>Per scene instance</summary>' + scene_table(D) + '</details>')
    h.append('<h2 id="task1">Task 1: the MPPI step</h2>')
    h.append(T.get('task1', ''))
    h.append(ladder(D, 'maze22_1'))
    h.append('<details><summary>Every optimization step on four scenes (maze 10 × 10, maze 22 × 22, gap ×4, house Onaga)</summary>' + ladder_table(D) + '</details>')
    h.append('<h3>Threads</h3>')
    h.append(T.get('threads', ''))
    h.append(threads_chart(D))
    h.append('<h3>Fewer, longer steps</h3>')
    h.append(T.get('knobs', ''))
    h.append(legend([('one-way', 'var(--s3)'), ('bidirectional', 'var(--s4)')]))
    h.append(knob_charts(D))
    h.append('<details><summary>Step multiplier, sample count and bidirectional variants: full table</summary>' + knob_table(D) + '</details>')
    h.append('<h2 id="task2">Task 2: a learned clearance field</h2>')
    h.append(T.get('task2', ''))
    h.append(dist_section(D))
    h.append('<h2 id="task3">Task 3: bidirectional MPPI</h2>')
    h.append(T.get('task3', ''))
    h.append(bidir_section(D))
    h.append(T.get('direction', ''))
    h.append(direction_section(D))
    h.append('<h2 id="setup">How it was measured</h2><div class="answers">')
    for q, a in T.get('setup', []):
        h.append(f'<div><div class="q">{q}</div><div class="a">{a}</div></div>')
    h.append('</div>')
    h.append(f'<footer>jeffrey/lbs_fast: report.py → report/summary.json → page.py. {n_final} all-scene benchmark runs, {len(D["runs"])} registered runs in total.</footer>')
    h.append('</div>')
    h.append(f'<script>{JS}</script>')
    return '\n'.join(h)


def main():
    D = json.load(open(HERE / 'report/summary.json'))
    tp = HERE / 'page_text.json'
    T = json.load(open(tp)) if tp.exists() else {}
    (HERE / 'report/lbs_fast.html').write_text(body(D, T))
    print(HERE / 'report/lbs_fast.html')


if __name__ == '__main__':
    main()
