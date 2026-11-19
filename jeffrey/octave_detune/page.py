"""Findings page of the octave-detune study: report/octave_detune.html (inline SVG charts, theme-aware, hover tips).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/report.py      # report/summary.json first
    ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/page.py

Numbers come from report/summary.json (report.py), report/spatial.json (spatial.py) and report/stalls.json (stalls.py);
the words from page_text.py.
"""
import html
import json
import math
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as FT  # noqa: E402
import page_text as TX  # noqa: E402
import svgcharts as C  # noqa: E402

OUT = HERE / 'report'
SIZES = ['13', '22']
SHIFT_ORDER = ['shift0.5', 'shift0.707', 'shift0.841', 'shift0.9', 'shift0.95', 'octave', 'shift1.05', 'shift1.1', 'shift1.189', 'shift1.414', 'shift2']
JITTER = ['jitter0.05', 'jitter0.1', 'jitter0.2']
PAIRED_ROWS = ['shift0.5', 'shift0.707', 'shift0.841', 'shift0.9', 'shift0.95', 'shift0.9999', 'shift1.0001', 'shift1.05', 'shift1.1', 'shift1.189',
               'shift1.414', 'shift2', 'jitter0.05', 'jitter0.1', 'jitter0.2', 'jitter0.4142', 'top0.9', 'rest0.9', 'octaveperm', 'octavecircle']
CLIP = 16.0          # paired charts: points; larger changes are drawn at the axis edge with their value written out
FULL_CURVE = {'1000', '2000', '3000', '4000', '5000'}
SHORT = {'octave': 'exact octave', 'shift1.0001': 'null ×1.0001', 'shift0.9999': 'null ×0.9999', 'shift0.95': 'shift ×0.95', 'shift1.05': 'shift ×1.05', 'shift0.9': 'shift ×0.9',
         'shift1.1': 'shift ×1.1', 'jitter0.05': 'jitter ±5 %', 'jitter0.1': 'jitter ±10 %', 'jitter0.2': 'jitter ±20 %', 'jitter0.4142': 'fill ±½ octave', 'shift0.841': 'shift ×0.84',
         'shift1.189': 'shift ×1.19', 'shift0.707': 'shift ×0.71', 'shift1.414': 'shift ×1.41', 'shift2': 'shift ×2 (locked)', 'shift0.5': 'shift ×0.5 (locked)',
         'top0.9': 'top octave ×0.9', 'rest0.9': 'all but top ×0.9',
         'octaveperm': 'octaveperm', 'octavecircle': 'octavecircle', 'B x 3': 'B × 3', 'B x 6': 'B × 6', 'ntrl-demo': 'ntrl-demo', 'gs:octave': 'grid-sync octave'}
BEST_B = {'13': 'B x 3', '22': 'B x 6'}
esc = C.esc


def field(S, key, n):
    return S['fields'].get(key, {}).get(n)


def kind_cls(key):
    """Series colour by what a variant does, the same on every chart (validated all-pairs, light and dark): the exact
    frequencies (octave and nulls) ink, every row lower orange, higher blue, jitter / fill teal, part of the band lower
    hollow orange, grid-sync controls and earlier fields dark gray (always labelled)."""
    if key in ('octave', 'gs:octave', 'shift1.0001', 'shift0.9999'):
        return 'k-exact'
    if key.startswith('shift'):
        return 'k-low' if float(key[5:]) < 1 else 'k-high'
    if key.startswith('jitter'):
        return 'k-jit'
    if key.startswith(('top', 'rest')):
        return 'k-part'
    return 'k-ref'


def tag(key):
    """Short direct label for scatter points."""
    if key in ('shift1.0001', 'shift0.9999'):
        return 'null'
    if key.startswith('shift'):
        return f'×{float(key[5:]):g}'
    if key == 'jitter0.4142':
        return 'fill'
    if key.startswith('jitter'):
        return f'±{100 * float(key[6:]):g} %'
    if key.startswith('top'):
        return 'top ×0.9'
    if key.startswith('rest'):
        return 'rest ×0.9'
    return dict(octaveperm='perm', octavecircle='circle').get(key, key)


def spread_labels(items, gap=11.0, char=5.8):
    """items = [(x px, y px, text)], left-anchored labels: push a label down until its box (about char px per character)
    overlaps no label already placed; returns [(x, y, text)]."""
    out = []
    for x, y, t in sorted(items, key=lambda i: i[1]):
        for _ in range(30):
            hit = [oy for ox, oy, ot in out if x < ox + char * len(ot) + 4 and ox < x + char * len(t) + 4 and abs(oy - y) < gap]
            if not hit:
                break
            y = max(hit) + gap
        out.append((x, y, t))
    return out


def factor_x(key):
    return 0.0 if key == 'octave' else math.log2(float(key[5:]))


SCAN_TICKS = [-1, -0.5, 0, 0.5, 1]
SCAN_LAB = {-1: '×0.5', -0.5: '×0.71', 0: '×1', 0.5: '×1.41', 1: '×2'}


def pct(v, p=1):
    return '–' if v is None else f'{100 * v:.{p}f} %'


# ---------------------------------------------------------------- chart A: the rows' frequencies
def chart_freq(n=22, seed=87358):
    specs = ['octave', 'shift0.9', 'shift1.1', 'jitter0.1', 'jitter0.4142', 'shift0.707', 'shift1.414', 'shift0.5', 'shift2', 'top0.9', 'rest0.9']
    E = n + 0.1
    B = FT.draw(seed)
    rowh, top = 26, 10
    lo = min(math.log2(float(np.abs(FT.transform(B, E, s).double().numpy()[:, :2]).max(1).min() / (E / 2))) for s in specs) - 0.4
    f = C.Frame(lo, 1.6, 0, 1, w=720, h=top + rowh * len(specs) + 44, left=150, right=16, top=top, bottom=40)
    body = []
    Ls = FT.octaves(n)
    Lgrid = [L for L in [2 * Ls[0]] + Ls + [0.5] if f.x0 <= math.log2(1 / L) <= f.x1]
    for L in Lgrid:
        x = f.X(math.log2(1 / L))
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{f.h - f.b}"/>')
    for i, spec in enumerate(specs):
        y = top + i * rowh + rowh / 2
        m = FT.transform(B, E, spec).double().numpy()
        d = np.abs(m[:, :2]).max(1) / (E / 2)
        cls = kind_cls(spec)
        body.append(f'<text class="ax" x="{f.l - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(SHORT[spec])}</text>')
        for v in d:
            x = f.X(math.log2(v))
            body.append(f'<line class="{cls} tick" x1="{x:.1f}" x2="{x:.1f}" y1="{y - 8:.1f}" y2="{y + 8:.1f}"/>')
        desc = FT.describe(FT.transform(B, E, spec), E, n, spec, B)
        tip = (f'{SHORT[spec]}: top octave edge phase error {desc["edge_phase_error_top_octave_median"]:.2f} rad, '
               f'rows locked {desc["locked_rows"]}/128')
        body.append(f'<rect class="hit" data-tip="{esc(tip)}" tabindex="0" x="{f.l}" y="{y - rowh / 2:.1f}" width="{f.w - f.l - f.r}" height="{rowh}"/>')
    yb = f.h - f.b
    body.append(f'<line class="base" x1="{f.l}" x2="{f.w - f.r}" y1="{yb}" y2="{yb}"/>')
    for L in Lgrid:
        x = f.X(math.log2(1 / L))
        lab = f'{L:g}' if L >= 1 else '½'
        body.append(f'<text class="ax" x="{x:.1f}" y="{yb + 16}" text-anchor="middle">{lab}</text>')
    body.append(f'<text class="ax axt" x="{(f.l + f.w - f.r) / 2:.1f}" y="{f.h - 4}" text-anchor="middle">'
                f'half period along the grid axis (cells), log scale; hairlines = the octave positions</text>')
    return C.svg(f, body, f'Dominant frequency of the 128 Fourier rows on the {n}x{n} maze, training seed {seed}, per variant')


# ---------------------------------------------------------------- chart B: success vs coherent shift
def chart_scan(S, n):
    pts, seeds_pts = [], []
    for key in SHIFT_ORDER:
        v = field(S, key, n)
        if v is None:
            continue
        x = factor_x(key)
        pts.append((x, v['pooled']['rate'], key, v))
        for s, k in v['pooled']['per_seed'].items():
            seeds_pts.append((x, k / (v['pooled']['runs'] / len(v['pooled']['per_seed'])), key, s))
    refs = [(k, field(S, k, n)) for k in ('octaveperm', 'octavecircle', BEST_B[n], 'ntrl-demo')]
    refs = [(k, v) for k, v in refs if v is not None]
    core = [p[1] for p in pts] + [p[1] for p in seeds_pts]
    near = [v['pooled']['rate'] for _, v in refs if core and v['pooled']['rate'] >= min(core) - 0.08]
    vals = core + near
    if not vals:
        return '<p class="empty">No runs yet.</p>'
    lo = max(0.0, math.floor((min(vals) - 0.04) * 20) / 20)
    hi = min(1.0, math.ceil((max(vals) + 0.03) * 20) / 20)
    f = C.Frame(-1.12, 1.1, lo, hi, w=460, h=300, left=46, right=104, top=14, bottom=46)
    body = f.axes(SCAN_TICKS, C.nice_ticks(lo, hi, 5), xlabel='frequency factor on every row (log scale)',
                  xfmt=lambda v: SCAN_LAB.get(v, f'{v:g}'), yfmt=lambda v: f'{100 * v:.0f} %')
    for x in (-1, 0, 1):
        body.append(f'<line class="lock" x1="{f.X(x):.1f}" x2="{f.X(x):.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="lbl small" x="{f.X(x) + 4:.1f}" y="{f.t + 10}">locked</text>')
    # reference rules across the plot, their labels in the right margin, spread apart where they would collide
    shown = sorted([(f.Y(v['pooled']['rate']), k, v) for k, v in refs if lo <= v['pooled']['rate'] <= hi])
    placed = []
    for y, k, v in shown:
        ly = max(y, placed[-1] + 12) if placed else y
        placed.append(ly)
    shift = max(0.0, (placed[-1] - (f.h - f.b)) if placed else 0.0)
    for (y, k, v), ly in zip(shown, placed):
        ly -= shift
        r = v['pooled']['rate']
        xe = f.w - f.r
        body.append(f'<line class="ref" x1="{f.l}" x2="{xe}" y1="{y:.1f}" y2="{y:.1f}"/>')
        body.append(f'<polyline class="ref" fill="none" points="{xe},{y:.1f} {xe + 5},{ly:.1f} {xe + 8},{ly:.1f}"/>')
        body.append(f'<text class="lbl small" x="{xe + 10}" y="{ly + 3.5:.1f}">{esc(SHORT[k])} {100 * r:.0f} %</text>')
        p = v['pooled']
        tip = f'{SHORT[k]}: {100 * r:.1f} % ({p["success"]}/{p["runs"]})'
        body.append(f'<rect class="hit" data-tip="{esc(tip)}" tabindex="0" x="{f.l}" y="{y - 5:.1f}" width="{xe - f.l}" height="10"/>')
    body.append(C.polyline(f, [(p[0], p[1]) for p in pts], 'ref', width=1.5))
    for x, y, key, s in seeds_pts:
        body.append(f'<circle class="{kind_cls(key)} seed" cx="{f.X(x):.1f}" cy="{f.Y(y):.1f}" r="2.6"/>')
    for x, y, key, v in pts:
        p = v['pooled']
        tip = (f'{SHORT[key]}: {100 * y:.1f} % ({p["success"]}/{p["runs"]}), seeds {" / ".join(map(str, p["per_seed"].values()))}'
               + (f'; vs the exact frequencies {v["diff_mean_points"]:+.1f} ± {v["diff_se_points"] or 0:.1f} pts' if v.get('diff_mean_points') is not None else ''))
        body.append(C.dot(f, x, y, kind_cls(key), r=4.5, tip=tip))
    below = [(k, v) for k, v in refs if v['pooled']['rate'] < lo]
    if below:
        body.append(f'<text class="lbl small" x="{f.l + 4}" y="{f.h - f.b - 6}">below the axis: '
                    + esc(', '.join(f'{SHORT[k]} {100 * v["pooled"]["rate"]:.0f} %' for k, v in below)) + '</text>')
    return C.svg(f, body, f'Success against the frequency factor, {n}x{n}')


# ---------------------------------------------------------------- chart B2: the two kinds of failure along the scan
FAIL_SERIES = [  # (measure, label, dash, dot style, x offset so the two whiskers at one factor do not overlap)
    ('wall_rate', 'through a wall', None, '', -0.025),
    ('never_arrived_rate', 'never arrive', '5 4', ' hollow', 0.025),
]


def chart_fail(S, n):
    """Change in the two kinds of failure against the coherent frequency factor, paired by seed against the exact-frequency
    trainings: runs that reach the goal only after crossing a wall, and runs that never reach it. Filled dots on a solid line
    = wall crossings, open dots on a dashed line = never arrived; dot colour = side of x1; whiskers +-2 standard errors."""
    rows = []
    for key in SHIFT_ORDER:
        v = field(S, key, n)
        if key == 'octave':
            rows.append((0.0, key, None))
        elif v and all(m in (v.get('paired_metrics') or {}) for m, *_ in FAIL_SERIES):
            rows.append((factor_x(key), key, v))
    if len(rows) < 3:
        return '<p class="empty">Not enough scan runs yet.</p>'
    vals = [0.0]
    for _, _, v in rows:
        for m, *_ in FAIL_SERIES:
            if v is not None:
                d = v['paired_metrics'][m]
                vals += [100 * (d['mean'] - 2 * (d.get('se') or 0)), 100 * (d['mean'] + 2 * (d.get('se') or 0))]
    lo, hi = min(vals) - 1.0, max(vals) + 1.0
    f = C.Frame(-1.12, 1.1, lo, hi, w=460, h=300, left=46, right=86, top=14, bottom=46)
    body = f.axes(SCAN_TICKS, C.nice_ticks(lo, hi, 6), xlabel='frequency factor on every row (log scale)',
                  xfmt=lambda v: SCAN_LAB.get(v, f'{v:g}'), yfmt=lambda v: '0' if abs(v) < 1e-9 else f'{v:+g}')
    body.append(f'<line class="zero" x1="{f.l}" x2="{f.w - f.r}" y1="{f.Y(0):.1f}" y2="{f.Y(0):.1f}"/>')
    for x in (-1, 0, 1):
        body.append(f'<line class="lock" x1="{f.X(x):.1f}" x2="{f.X(x):.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="lbl small" x="{f.X(x) + 4:.1f}" y="{f.h - f.b - 5}">locked</text>')
    o = field(S, 'octave', n) or {}
    ends = []
    for m, lab, dash, hollow, dx in FAIL_SERIES:
        pts = [(x + dx, 0.0 if v is None else 100 * v['paired_metrics'][m]['mean']) for x, _, v in rows]
        body.append(C.polyline(f, pts, 'k-ref', width=1.6, dash=dash))
        ends.append((f.X(pts[-1][0]) + 8, f.Y(pts[-1][1]) + 3.5, lab))
    for m, lab, dash, hollow, dx in FAIL_SERIES:
        for x, key, v in rows:
            if v is None:
                tip = f'exact octave, runs that {lab}: {pct(o.get(m))} of the runs (the reference, change 0)'
                body.append(C.dot(f, x + dx, 0.0, 'k-exact' + hollow, r=4 if hollow else 5, tip=tip))
                continue
            d = v['paired_metrics'][m]
            y, se = 100 * d['mean'], 100 * (d.get('se') or 0)
            if se:
                body.append(f'<line class="{kind_cls(key)} whisk" x1="{f.X(x + dx):.1f}" x2="{f.X(x + dx):.1f}" '
                            f'y1="{f.Y(y - 2 * se):.1f}" y2="{f.Y(y + 2 * se):.1f}"/>')
            tip = (f'{SHORT[key]}, runs that {lab}: {y:+.1f} ± {se:.1f} points over {d["k"]} seeds'
                   + (f' (z {d["z"]:+.1f})' if d.get('z') is not None else '') + f'; {pct(v.get(m))} of the runs, exact octave {pct(o.get(m))}')
            body.append(C.dot(f, x + dx, y, kind_cls(key) + hollow, r=4 if hollow else 5, tip=tip))
    for lx, ly, t in spread_labels(ends, 12):
        body.append(f'<text class="lbl small" x="{lx:.1f}" y="{ly:.1f}">{esc(t)}</text>')
    return C.svg(f, body, f'Change in wall crossings and in runs that never arrive against the frequency factor, {n}x{n}')


SCAN_METRICS = {  # key -> (label, unit, value of a paired mean d, its absolute value for the octave tooltip)
    'loss_final_mean100': ('final training loss', '%', lambda d: 100 * (math.exp(d) - 1), lambda v: f'{v:.4f}'),
    'loss_settle_epoch': ('loss settling epoch', 'epochs', lambda d: d, lambda v: f'{v:,.0f}'),
}


def chart_scan_metric(S, n, m):
    """Paired change of one measure against the coherent frequency factor: the exact octave at 0, every shift the mean over
    seeds of its change against the same seed's exact-frequency trainings, whiskers +-2 standard errors (mapped through the
    unit, so a percent change of a log-ratio has asymmetric whiskers)."""
    lab, unit, conv, absf = SCAN_METRICS[m]
    rows = []
    for key in SHIFT_ORDER:
        v = field(S, key, n)
        if key == 'octave':
            rows.append((0.0, key, None))
        elif v and m in (v.get('paired_metrics') or {}):
            rows.append((factor_x(key), key, v))
    if len(rows) < 3:
        return '<p class="empty">Not enough scan runs yet.</p>'
    pts, vals = [], [0.0]
    for x, key, v in rows:
        if v is None:
            pts.append((x, key, 0.0, None, None, None))
            continue
        d = v['paired_metrics'][m]
        se = d.get('se') or 0
        y, ylo, yhi = conv(d['mean']), conv(d['mean'] - 2 * se), conv(d['mean'] + 2 * se)
        pts.append((x, key, y, ylo, yhi, d))
        vals += [ylo, yhi]
    span = max(vals) - min(vals)
    lo, hi = min(vals) - 0.04 * span, max(vals) + 0.04 * span
    f = C.Frame(-1.12, 1.1, lo, hi, w=460, h=260, left=58, right=16, top=14, bottom=46)
    yfmt = (lambda v: '0' if abs(v) < 1e-9 else f'{v:+,.0f}') if unit == 'epochs' else (lambda v: '0' if abs(v) < 1e-9 else f'{v:+g} %')
    body = f.axes(SCAN_TICKS, C.nice_ticks(lo, hi, 6), xlabel='frequency factor on every row (log scale)',
                  xfmt=lambda v: SCAN_LAB.get(v, f'{v:g}'), yfmt=yfmt)
    body.append(f'<line class="zero" x1="{f.l}" x2="{f.w - f.r}" y1="{f.Y(0):.1f}" y2="{f.Y(0):.1f}"/>')
    for x in (-1, 0, 1):
        body.append(f'<line class="lock" x1="{f.X(x):.1f}" x2="{f.X(x):.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="lbl small" x="{f.X(x) + 4:.1f}" y="{f.t + 10}">locked</text>')
    body.append(C.polyline(f, [(x, y) for x, _, y, *_ in pts], 'k-ref', width=1.6))
    o = field(S, 'octave', n) or {}
    for x, key, y, ylo, yhi, d in pts:
        if d is None:
            ov = o.get(m)
            tip = f'exact octave: {lab} {absf(ov) if ov is not None else "–"} (the reference, change 0)'
            body.append(C.dot(f, x, 0.0, 'k-exact', r=5, tip=tip))
            continue
        body.append(f'<line class="{kind_cls(key)} whisk" x1="{f.X(x):.1f}" x2="{f.X(x):.1f}" y1="{f.Y(ylo):.1f}" y2="{f.Y(yhi):.1f}"/>')
        q = (lambda t: f'{t:+,.0f}') if unit == 'epochs' else (lambda t: f'{t:+.1f}')
        tip = (f'{SHORT[key]}, {lab}: {q(y)} {unit} over {d["k"]} seeds' + (f' (z {d["z"]:+.1f})' if d.get('z') is not None else '')
               + f'; ±2 SE {q(ylo)} … {q(yhi)} {unit}')
        body.append(C.dot(f, x, y, kind_cls(key), r=5, tip=tip))
    return C.svg(f, body, f'Change in {lab} against the frequency factor, {n}x{n}')


# ---------------------------------------------------------------- where the runs that never arrive stop (planning seed 0)
STALL_REF = ('octave', 'shift1.0001', 'shift0.9999')
STALL_KEYS = ('runs', 'success', 'through_wall', 'stop_wall', 'stop_open', 'moving', 'stop_wall_route', 'stop_open_route')


def stall_groups(n):
    """{spec: summed counts over its runs} from report/stalls.json (stalls.py); the exact frequencies (octave and both
    nulls) pooled under 'octave'."""
    sp = json.loads((OUT / 'stalls.json').read_text()) if (OUT / 'stalls.json').exists() else {}
    out = {}
    for r in sp.values():
        if r['n'] != n:
            continue
        a = out.setdefault('octave' if r['spec'] in STALL_REF else r['spec'], dict(seeds=set(), trainings=0))
        a['seeds'].add(r['seed'])
        a['trainings'] += 1
        for k in STALL_KEYS:
            a[k] = a.get(k, 0) + r[k]
    return out


def chart_stall(n):
    """Runs that never arrive, by where they stop: against a wall (filled) or in the open corridor (open dot), as a share of
    all planning-seed-0 runs of the variant, one row per coherent shift."""
    G = stall_groups(n)
    rows = [k for k in SHIFT_ORDER if k in G]
    if 'octave' not in G or len(rows) < 3:
        return '<p class="empty">No stall record yet.</p>'
    share = lambda a, k: 100 * a[k] / a['runs']  # noqa: E731
    hi = max(max(share(G[k], 'stop_wall'), share(G[k], 'stop_open')) for k in rows)
    hi = math.ceil((hi + 1.5) / 5) * 5
    rowh = 22
    f = C.Frame(0, hi, 0, 1, w=460, h=12 + rowh * len(rows) + 42, left=128, right=16, top=10, bottom=42)
    body = []
    for t in C.nice_ticks(0, hi, 5):
        x = f.X(t)
        body.append(f'<line class="{"zero" if t == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="ax" x="{x:.1f}" y="{f.h - f.b + 16}" text-anchor="middle">{t:g} %</text>')
    body.append(f'<text class="ax axt" x="{(f.l + f.w - f.r) / 2:.1f}" y="{f.h - 4}" text-anchor="middle">share of all runs (planning seed 0)</text>')
    for i, k in enumerate(rows):
        a = G[k]
        y = f.t + i * rowh + rowh / 2
        cls = kind_cls(k)
        lab = 'exact (octave + nulls)' if k == 'octave' else SHORT[k]
        body.append(f'<text class="ax" x="{f.l - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(lab)}</text>')
        w, o = share(a, 'stop_wall'), share(a, 'stop_open')
        body.append(f'<line class="grid" x1="{f.X(min(w, o)):.1f}" x2="{f.X(max(w, o)):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke-width="2"/>')
        base = f'{lab}, {a["runs"]} runs from {a["trainings"]} trainings'
        for key, val, hollow, what in (('stop_wall', w, '', 'stopped against a wall'), ('stop_open', o, ' hollow', 'stopped in the open corridor')):
            tip = (f'{base}: {val:.1f} % {what} ({a[key]} runs, {a[key + "_route"]} of them on the true route); '
                   f'reached the goal {share(a, "success"):.1f} %, through a wall {share(a, "through_wall"):.1f} %, still moving {share(a, "moving"):.1f} %')
            cx = f.X(val)
            body.append(f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><circle class="hit" cx="{cx:.1f}" cy="{y:.1f}" r="10"/>'
                        f'<circle class="{cls}{hollow} dot ring" cx="{cx:.1f}" cy="{y:.1f}" r="{4.5 if hollow else 5.5}"/></g>')
    return C.svg(f, body, f'Where the runs that never arrive stop, by frequency factor, {n}x{n}')


# ---------------------------------------------------------------- chart C: paired differences
def chart_paired(S, n):
    rows = [k for k in PAIRED_ROWS if field(S, k, n) and field(S, k, n).get('paired')]
    if not rows:
        return '<p class="empty">No paired runs yet.</p>'
    nz = S['noise'].get(n) or {}
    sd = nz.get('sd_run_points')
    band = 2 * sd * math.sqrt(4 / 3) if sd else None           # one seed's difference to the mean of ~3 reference runs
    vals = []
    for k in rows:
        v = field(S, k, n)
        vals += [100 * p['diff'] for p in v['paired'].values()]
        if v.get('diff_se_points'):
            vals += [v['diff_mean_points'] - 2 * v['diff_se_points'], v['diff_mean_points'] + 2 * v['diff_se_points']]
    lo = max(min(min(vals), -(band or 2)) - 0.5, -CLIP)
    hi = min(max(max(vals), band or 2) + 0.5, CLIP)
    cx = lambda v: min(max(v, lo), hi)  # noqa: E731
    rowh = 24
    f = C.Frame(lo, hi, 0, 1, w=460, h=16 + rowh * len(rows) + 40, left=112, right=14, top=12, bottom=40)
    body = []
    if band:
        x0, x1 = f.X(-band), f.X(band)
        body.append(f'<rect class="band" x="{x0:.1f}" y="{f.t}" width="{x1 - x0:.1f}" height="{f.h - f.t - f.b}"/>')
    for t in C.nice_ticks(lo, hi, 6):
        x = f.X(t)
        body.append(f'<line class="{"zero" if t == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="ax" x="{x:.1f}" y="{f.h - f.b + 16}" text-anchor="middle">{"0" if t == 0 else f"{t:+g}"}</text>')
    body.append(f'<text class="ax axt" x="{(f.l + f.w - f.r) / 2:.1f}" y="{f.h - 4}" text-anchor="middle">success minus the exact-frequency trainings of the same seed (points)</text>')
    for i, k in enumerate(rows):
        v = field(S, k, n)
        y = f.t + i * rowh + rowh / 2
        cls = kind_cls(k)
        body.append(f'<text class="ax" x="{f.l - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(SHORT[k])}</text>')
        for s, p in v['paired'].items():
            body.append(f'<circle class="{cls} seed" cx="{f.X(cx(100 * p["diff"])):.1f}" cy="{y:.1f}" r="3"/>')
        m, se = v['diff_mean_points'], v.get('diff_se_points')
        if se:
            body.append(f'<line class="{cls} whisk" x1="{f.X(cx(m - 2 * se)):.1f}" x2="{f.X(cx(m + 2 * se)):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        if m < lo or m > hi:
            side = m < lo
            tx = f.X(lo) + 8 if side else f.X(hi) - 8
            body.append(f'<text class="lbl small" x="{tx:.1f}" y="{y - 6:.1f}" text-anchor="{"start" if side else "end"}">'
                        f'{"← " if side else ""}{m:+.1f}{"" if side else " →"} (off the axis)</text>')
        z = v.get('diff_z')
        tip = (f'{SHORT[k]}: mean {m:+.1f} pts' + (f' ± {se:.1f} (z {z:+.1f})' if se else '') + f' over {len(v["paired"])} seeds ('
               + ', '.join(f'{s}: {100 * p["diff"]:+.1f}' for s, p in v['paired'].items()) + f'); runs gained {v["gained"]}, lost {v["lost"]} vs the exact octave run')
        xm = f.X(cx(m))
        body.append(f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><rect class="hit" x="{f.l}" y="{y - rowh / 2:.1f}" width="{f.w - f.l - f.r}" height="{rowh}"/>'
                    f'<line class="{cls} meanbar" x1="{xm:.1f}" x2="{xm:.1f}" y1="{y - 8:.1f}" y2="{y + 8:.1f}"/></g>')
    return C.svg(f, body, f'Paired success difference to the exact-frequency trainings, {n}x{n}')


# ---------------------------------------------------------------- which part of the band carries the x0.9 loss
BAND_ROWS = [('shift0.9', 'every row ×0.9'), ('top0.9', 'top octave only ×0.9'), ('rest0.9', 'all but the top ×0.9')]
BAND_PANELS = [  # (key, measure, axis label, transform of a paired change, value format)
    ('success', 'success_rate', 'success change (points)', lambda d: 100 * d, lambda v: f'{v:+.1f}'),
    ('wall', 'wall_rate', 'change in runs through a wall (points)', lambda d: 100 * d, lambda v: f'{v:+.1f}'),
    ('loss', 'loss_final_mean100', 'final training loss change (%)', lambda d: 100 * (math.exp(d) - 1), lambda v: f'{v:+.0f} %'),
    ('corr', 'speed_corr', 'held-out speed correlation change', lambda d: d, lambda v: f'{v:+.3f}'),
]


def chart_band(S, n, measure, xlabel, tf, vfmt):
    """One measure, the three ways of lowering part of the band by 10 %: seeds (small dots), mean (bar) and +-2 standard
    errors from training noise (whisker), all as changes against the same seed's exact-frequency trainings."""
    rows = [(k, lab, field(S, k, n)['paired_metrics'][measure]) for k, lab in BAND_ROWS
            if field(S, k, n) and measure in (field(S, k, n).get('paired_metrics') or {})]
    if len(rows) < 2:
        return '<p class="empty">The part-of-band runs are not in yet.</p>'
    vals = [0.0]
    for _, _, d in rows:
        vals += [tf(x) for x in d['deltas'].values()]
        if d.get('se'):
            vals += [tf(d['mean'] - 2 * d['se']), tf(d['mean'] + 2 * d['se'])]
    span = max(vals) - min(vals)
    lo, hi = min(vals) - 0.08 * span, max(vals) + 0.08 * span
    rowh = 28
    f = C.Frame(lo, hi, 0, 1, w=460, h=14 + rowh * len(rows) + 42, left=142, right=16, top=10, bottom=42)
    body = []
    for t in C.nice_ticks(lo, hi, 5):
        x = f.X(t)
        body.append(f'<line class="{"zero" if abs(t) < 1e-12 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="ax" x="{x:.1f}" y="{f.h - f.b + 16}" text-anchor="middle">{"0" if abs(t) < 1e-12 else f"{t:+g}"}</text>')
    body.append(f'<text class="ax axt" x="{(f.l + f.w - f.r) / 2:.1f}" y="{f.h - 4}" text-anchor="middle">{esc(xlabel)}</text>')
    for i, (k, lab, d) in enumerate(rows):
        y = f.t + i * rowh + rowh / 2
        cls = kind_cls(k)
        body.append(f'<text class="ax" x="{f.l - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(lab)}</text>')
        for s, x in d['deltas'].items():
            body.append(f'<circle class="{cls} seed" cx="{f.X(tf(x)):.1f}" cy="{y:.1f}" r="3"/>')
        m, se = d['mean'], d.get('se')
        if se:
            body.append(f'<line class="{cls} whisk" x1="{f.X(tf(m - 2 * se)):.1f}" x2="{f.X(tf(m + 2 * se)):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        tip = (f'{lab}: {vfmt(tf(m))}' + (f' (z {d["z"]:+.1f})' if d.get('z') is not None else '') + ' over seeds '
               + ', '.join(f'{s}: {vfmt(tf(x))}' for s, x in d['deltas'].items()))
        xm = f.X(tf(m))
        body.append(f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><rect class="hit" x="{f.l}" y="{y - rowh / 2:.1f}" width="{f.w - f.l - f.r}" height="{rowh}"/>'
                    f'<line class="{cls} meanbar" x1="{xm:.1f}" x2="{xm:.1f}" y1="{y - 8:.1f}" y2="{y + 8:.1f}"/></g>')
        lx = min(xm + 10, f.w - f.r - 4)
        body.append(f'<text class="lbl small" x="{lx:.1f}" y="{y - 7:.1f}">{esc(vfmt(tf(m)))}</text>')
    return C.svg(f, body, f'{xlabel}: every row x0.9, top octave only x0.9, all but the top x0.9, {n}x{n}')


# ---------------------------------------------------------------- lock error against the paired difference
def lock_error(key, n, seeds):
    """Median top-octave edge phase error (rad) of a variant's matrix, mean over its training seeds; the grid-sync
    controls are rebuilt with jeffrey/grid_sync/features.py."""
    import importlib.util
    E, nn = float(n) + 0.1, int(n)
    vals = []
    for s in seeds:
        B = FT.draw(int(s))
        if key in ('octaveperm', 'octavecircle'):
            spec_ = importlib.util.spec_from_file_location('gs_features', HERE.parent / 'grid_sync/features.py')
            gs = importlib.util.module_from_spec(spec_)
            spec_.loader.exec_module(gs)
            m = gs.transform(B, E, key)
        else:
            m = FT.transform(B, E, key)
        d = FT.describe(m, E, nn, key)
        vals.append(d['edge_phase_error_top_octave_median'])
    return float(np.mean(vals))


WIDE = ('shift0.707', 'shift0.841', 'shift1.189', 'shift1.414')     # the wide scan moves the level by a quarter octave or more


def chart_lock(S, n):
    pts = []
    for k in PAIRED_ROWS:
        if k in WIDE:          # its lock error (several rad, wrapping) says nothing next to its change of level; see the scan
            continue
        v = field(S, k, n)
        if not v or v.get('diff_mean_points') is None:
            continue
        pts.append((lock_error(k, n, v['seeds']), v['diff_mean_points'], k, v))
    if not pts:
        return '<p class="empty">No paired runs yet.</p>'
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ses = [p[3].get('diff_se_points') or 0 for p in pts]
    sd = float(np.median(ses)) if ses else None                 # typical standard error of a variant's mean
    lo = max(min(min(y - 2 * e for y, e in zip(ys, ses)), -(2 * sd if sd else 1)) - 0.5, -CLIP)
    hi = min(max(max(y + 2 * e for y, e in zip(ys, ses)), 2 * sd if sd else 1) + 0.5, CLIP)
    cy = lambda v: min(max(v, lo), hi)  # noqa: E731
    xmax = max(xs) * 1.08 + 0.2
    f = C.Frame(0, xmax, lo, hi, w=460, h=290, left=48, right=14, top=14, bottom=44)
    body = f.axes(C.nice_ticks(0, xmax, 6), C.nice_ticks(lo, hi, 6), xlabel='top-octave phase error at the maze edge (rad)',
                  xfmt=lambda v: f'{v:g}', yfmt=lambda v: '0' if v == 0 else f'{v:+g}')
    if sd:
        body.insert(0, C.band(f, -2 * sd, 2 * sd))
    body.append(f'<line class="zero" x1="{f.l}" x2="{f.w - f.r}" y1="{f.Y(0):.1f}" y2="{f.Y(0):.1f}"/>')
    labels = []
    for x, y, k, v in pts:
        cls = kind_cls(k)
        se = v.get('diff_se_points')
        if se:
            body.append(f'<line class="{cls} whisk" x1="{f.X(x):.1f}" x2="{f.X(x):.1f}" y1="{f.Y(cy(y - 2 * se)):.1f}" y2="{f.Y(cy(y + 2 * se)):.1f}"/>')
        tip = f'{SHORT[k]}: phase error {x:.2f} rad, {y:+.1f} pts' + (f' ± {se:.1f}' if se else '') + f' vs the exact frequencies ({len(v["paired"])} seeds)'
        body.append(C.dot(f, x, cy(y), cls, r=4.5, tip=tip))
        labels.append((f.X(x) + 7, f.Y(cy(y)) + 3.5, tag(k) + (f' {y:+.0f} (off the axis)' if (y < lo or y > hi) else '')))
    for lx, ly, t in spread_labels(labels):
        body.append(f'<text class="lbl small" x="{lx:.1f}" y="{ly:.1f}">{esc(t)}</text>')
    return C.svg(f, body, f'Paired success difference against the lock error, {n}x{n}')


# ---------------------------------------------------------------- chart D/E: convergence
CONV_SERIES = [('octave', 'k-exact', 'exact octave', None), ('shift0.9', 'k-low', 'every row ×0.9', None), ('shift1.1', 'k-high', 'every row ×1.1', None),
               ('JITTER', 'k-jit', 'jitter ±5–20 % (mean)', None), ('BEST', 'k-ref', 'best B × s (ntrl-demo scaled)', '5 4'),
               ('ntrl-demo', 'k-ref', 'ntrl-demo', '1.5 3')]


def complete(c):
    return c and FULL_CURVE <= set(c)


BASE_SEEDS = ('87358', '1', '2')   # the training seeds every variant runs on both mazes


def base_curve(S, key, n):
    """A variant's checkpoint curve averaged over the base seeds only, so every line on a convergence chart is the same seed set
    (on 22x22 the exact octave, the nulls, x1.41 and x2 also have seeds 3 and 4). None until every base seed is in; a field
    from an earlier study with other seeds keeps its own mean."""
    v = field(S, key, n)
    if not v or not v.get('curve'):
        return None
    cs = v.get('curve_by_seed')
    if not cs or not any(s in cs for s in BASE_SEEDS):
        return v['curve']
    if not all(s in cs for s in BASE_SEEDS):
        return None
    epochs = set.intersection(*[set(cs[s]) for s in BASE_SEEDS])
    out = {}
    for e in epochs:
        out[e] = {}
        for m in cs[BASE_SEEDS[0]][e]:
            vals = [cs[s][e].get(m) for s in BASE_SEEDS]
            out[e][m] = float(np.mean(vals)) if all(x is not None for x in vals) else None
    return out


def base_loss_curve(S, key, n):
    """Median loss curve over the base seeds (or the field's own median for an earlier study)."""
    v = field(S, key, n)
    if not v:
        return None
    ls = v.get('loss_curve_by_seed')
    if not ls or not any(s in ls for s in BASE_SEEDS):
        return v.get('loss_curve_median')
    if not all(s in ls for s in BASE_SEEDS):
        return None
    return np.median(np.array([ls[s] for s in BASE_SEEDS]), 0).tolist()


def conv_curve(S, key, n, metric):
    if key == 'JITTER':
        curves = [base_curve(S, k, n) for k in JITTER]
        curves = [c for c in curves if complete(c)]
        if not curves:
            return []
        epochs = sorted(set.intersection(*[set(c) for c in curves]), key=int)
        return [(int(e), float(np.mean([c[e][metric] for c in curves]))) for e in epochs if all(c[e].get(metric) is not None for c in curves)]
    key = BEST_B[n] if key == 'BEST' else key
    c = base_curve(S, key, n)
    if not c:
        return []
    return sorted((int(e), x[metric]) for e, x in c.items() if x.get(metric) is not None)


def chart_conv(S, n, metric='success', ylab='success, planning seed 0', fmt=lambda v: f'{100 * v:.0f} %'):
    series = [(k, cls, lab, conv_curve(S, k, n, metric), dash) for k, cls, lab, dash in CONV_SERIES]
    series = [s for s in series if s[3]]
    if not series:
        return '<p class="empty">No convergence record yet.</p>'
    vals = [y for s in series for _, y in s[3]]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.05
    lo, hi = max(0.0, lo - pad), hi + pad
    if metric == 'success':
        hi = min(1.0, hi)
    f = C.Frame(0, 5000, lo, hi, w=460, h=280, left=48, right=14, top=14, bottom=42)
    body = f.axes([0, 1000, 2000, 3000, 4000, 5000], C.nice_ticks(lo, hi, 5), xlabel='training epoch', xfmt=lambda v: f'{v:,.0f}', yfmt=fmt)
    for key, cls, lab, pts, dash in series:
        body.append(C.polyline(f, pts, cls, dash=dash))
        for e, y in pts:
            body.append(C.dot(f, e, y, cls, r=3.5 if cls != 'k-ref' else 2.5, tip=f'{lab}, epoch {e:,}: {fmt(y)}'))
    return C.svg(f, body, f'{ylab} against training epoch, {n}x{n}')


def loss_series(S, key, n):
    if key == 'JITTER':
        cs = [c for c in (base_loss_curve(S, k, n) for k in JITTER) if c]
        return np.median(np.array(cs), 0).tolist() if cs else None
    key = BEST_B[n] if key == 'BEST' else key
    return base_loss_curve(S, key, n)


def chart_loss(S, n):
    series = [(k, cls, lab, loss_series(S, k, n), dash) for k, cls, lab, dash in CONV_SERIES]
    series = [s for s in series if s[3]]
    if not series:
        return '<p class="empty">No loss record yet.</p>'
    vals = [y for s in series for y in s[3][2:] if y and y > 0]
    lo, hi = min(vals) * 0.85, max(vals) * 1.15
    f = C.Frame(0, 5000, lo, hi, w=460, h=280, left=56, right=14, top=14, bottom=42, ylog=True)
    ticks = [t for t in [5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2] if lo <= t <= hi]
    body = f.axes([0, 1000, 2000, 3000, 4000, 5000], ticks, xlabel='training epoch (50-epoch means)', xfmt=lambda v: f'{v:,.0f}', yfmt=lambda v: f'{v:.0e}'.replace('e-0', 'e-'))
    for key, cls, lab, ys, dash in series:
        pts = [(25 + 50 * i, y) for i, y in enumerate(ys) if i >= 2 and y]
        body.append(C.polyline(f, pts, cls, dash=dash))
        e, y = pts[-1]
        body.append(C.dot(f, e, y, cls, r=3.5, tip=f'{lab}: loss {y:.2e} over the last 50 epochs'))
    return C.svg(f, body, f'Training loss against epoch, {n}x{n}')


EPOCH_SERIES = [('1000', 'q1', 'epoch 1,000'), ('2000', 'q2', 'epoch 2,000'), ('5000', 'q3', 'epoch 5,000')]


def chart_scan_epochs(S, n):
    """Success (planning seed 0) at three training epochs against the coherent frequency factor: how fast each shift converges."""
    rows = []
    for key in SHIFT_ORDER:
        c = base_curve(S, key, n)
        # only shifts with every base seed in, all averaged over the same seeds (fewer seeds would read as a dip)
        if c and all(e in c for e, _, _ in EPOCH_SERIES):
            rows.append((factor_x(key), key, dict(curve=c, curve_seeds=len(BASE_SEEDS))))
    if len(rows) < 2:
        return '<p class="empty">Not enough convergence records yet.</p>'
    vals = [v['curve'][e]['success'] for _, _, v in rows for e, _, _ in EPOCH_SERIES]
    lo = max(0.0, math.floor((min(vals) - 0.03) * 20) / 20)
    hi = min(1.0, math.ceil((max(vals) + 0.03) * 20) / 20)
    f = C.Frame(-1.12, 1.1, lo, hi, w=460, h=280, left=46, right=70, top=14, bottom=46)
    body = f.axes(SCAN_TICKS, C.nice_ticks(lo, hi, 5), xlabel='frequency factor on every row (log scale)',
                  xfmt=lambda v: SCAN_LAB.get(v, f'{v:g}'), yfmt=lambda v: f'{100 * v:.0f} %')
    ends = []
    for e, cls, lab in EPOCH_SERIES:
        pts = [(x, v['curve'][e]['success']) for x, _, v in rows]
        body.append(C.polyline(f, pts, cls))
        for (x, key, v), (_, y) in zip(rows, pts):
            body.append(C.dot(f, x, y, cls, r=3.5, tip=f'{SHORT[key]}, {lab}: {100 * y:.1f} % (planning seed 0, mean of {v["curve_seeds"]} seeds)'))
        ends.append((f.X(pts[-1][0]) + 8, f.Y(pts[-1][1]) + 3.5, lab[6:]))
    for lx, ly, t in spread_labels(ends, 12):
        body.append(f'<text class="lbl small" x="{lx:.1f}" y="{ly:.1f}">{esc(t)}</text>')
    return C.svg(f, body, f'Success at epochs 1,000, 2,000 and 5,000 against the frequency factor, {n}x{n}')


# ---------------------------------------------------------------- the change map
CHANGE_COLS = [('success_rate', 'success', 1, 'pts'), ('log_err', '|log T/T*|', -1, ''), ('speed_corr', 'speed fit', 1, ''),
               ('slope_2_8', 'wall slope', 1, ''), ('cos_2_8', 'wall cos', 1, ''), ('bumps', 'minima/path', -1, ''),
               ('never_arrived_rate', 'never arrived', -1, 'pts'), ('wall_rate', 'hit a wall', -1, 'pts'), ('time_ratio', 'time/truth', -1, ''),
               ('loss_final_mean100', 'final loss', -1, '%'), ('succ_at_1000', 'success @1k', 1, 'pts'), ('succ_at_2000', 'success @2k', 1, 'pts'),
               ('loss_settle_epoch', 'settle epoch', -1, ''), ('rollbacks', 'rollbacks', -1, '')]
ABS_KEY = dict(success_rate=None, never_arrived_rate='never_arrived_rate', wall_rate='wall_rate')


def change_map(S, n):
    """Mean paired change of every measure against the exact octave of the same seed; shade = |z| / 3, z = mean change over
    its standard error from training noise (per-seed SD of the null runs' changes / sqrt(seeds)); nulls themselves unshaded."""
    o = field(S, 'octave', n)
    if not o:
        return '<p class="empty">No exact-octave runs yet.</p>'
    msd = (S['noise'].get(n) or {}).get('metric_sd') or {}
    rows = [k for k in PAIRED_ROWS + ['B x 3', 'B x 6', 'ntrl-demo'] if field(S, k, n) and field(S, k, n).get('paired_metrics')]
    head = ''.join(f'<th scope="col">{esc(c[1])}</th>' for c in CHANGE_COLS)
    out = [f'<div class="scroll"><table class="cmap"><thead><tr><th scope="col">change vs exact octave, {n}×{n}</th><th scope="col">seeds</th>{head}</tr></thead><tbody>']
    ref = ''.join(f'<td class="refv">{fmt_abs(o, c)}</td>' for c in CHANGE_COLS)
    out.append(f'<tr class="refrow"><th scope="row">exact octave (value)</th><td>{len(o["seeds"])}</td>{ref}</tr>')
    for k in rows:
        v = field(S, k, n)
        pm = v['paired_metrics']
        cells = []
        for c in CHANGE_COLS:
            d = pm.get(c[0])
            if d is None:
                cells.append('<td>–</td>')
                continue
            sd = (msd.get(c[0]) or {}).get('sd')
            z = d['mean'] * c[2] / (sd / math.sqrt(d['k'])) if sd else 0.0
            shade = 0.0 if k in ('shift1.0001', 'shift0.9999') else max(-1.0, min(1.0, z / 3))
            cls = 'pos' if shade > 0.34 else ('neg' if shade < -0.34 else 'neu')
            tip = f'{SHORT.get(k, k)}, {c[1]}: change {fmt_delta(d["mean"], c)} over {d["k"]} seeds' + (f', {z:+.1f} noise SE' if sd else '')
            cells.append(f'<td class="{cls}" style="--z:{abs(shade):.2f}" data-tip="{esc(tip)}">{fmt_delta(d["mean"], c)}</td>')
        out.append(f'<tr><th scope="row">{esc(SHORT.get(k, k))}</th><td>{len(v["seeds"])}</td>{"".join(cells)}</tr>')
    out.append('</tbody></table></div>')
    return ''.join(out)


def fmt_abs(o, c):
    key = c[0]
    if key == 'success_rate':
        return pct(o['pooled']['rate'])
    v = o.get(key)
    if v is None:
        return '–'
    if key in ('never_arrived_rate', 'wall_rate', 'succ_at_1000', 'succ_at_2000'):
        return pct(v)
    if key == 'loss_final_mean100':
        return f'{v:.2e}'
    if key in ('loss_settle_epoch', 'rollbacks'):
        return f'{v:,.0f}'
    return f'{v:.3f}'


def fmt_delta(d, c):
    key = c[0]
    if key in ('success_rate', 'never_arrived_rate', 'wall_rate', 'succ_at_1000', 'succ_at_2000'):
        return f'{100 * d:+.1f}'
    if key == 'loss_final_mean100':
        return f'{100 * (math.exp(d) - 1):+.0f} %'
    if key in ('loss_settle_epoch', 'rollbacks'):
        return f'{d:+,.0f}'
    return f'{d:+.3f}'


# ---------------------------------------------------------------- spatial
SPATIAL_GROUPS = [('ring_down', 'k-low', 'shift ×0.9 / ×0.95'), ('ring_up', 'k-high', 'shift ×1.05 / ×1.1'), ('ring_jitter', 'k-jit', 'jitter ±5–20 %')]
RINGS = ['inner', 'middle', 'outer']
RING_METRICS = [f'slope_ring_{r}' for r in RINGS]


def chart_spatial(S, n):
    """Paired change of the near-wall slope by ring of the maze (inner / middle / outer third of the distance from the centre):
    mean over a group's (variant, seed) pairs of the change against the same seed's exact-frequency trainings (report.GROUPS
    ring_*), whiskers +-2 standard errors from training noise; the exact octave is the zero line."""
    data = []
    for k, (g, cls, lab) in enumerate(SPATIAL_GROUPS):
        d = ((S.get('groups') or {}).get(g) or {}).get(n) or {}
        if all(m in d for m in RING_METRICS):
            data.append((k, cls, lab, [d[m] for m in RING_METRICS], d.get('slope_outer_minus_inner')))
    if not data:
        return '<p class="empty">Spatial record not computed yet.</p>'
    vals = [0.0] + [x['mean'] + sg * 2 * x['se'] for *_, ds, _ in data for x in ds for sg in (-1, 1)]
    span = max(vals) - min(vals)
    lo, hi = min(vals) - 0.06 * span, max(vals) + 0.12 * span
    f = C.Frame(-0.35, 2.35, lo, hi, w=460, h=260, left=52, right=14, top=14, bottom=42)
    body = f.axes([0, 1, 2], C.nice_ticks(lo, hi, 5), xlabel='ring of the near-wall point (distance from the maze centre)',
                  xfmt=lambda v: ['inner third', 'middle third', 'outer third'][int(v)], yfmt=lambda v: '0' if abs(v) < 1e-9 else f'{v:+.2f}')
    body.append(f'<line class="zero" x1="{f.l}" x2="{f.w - f.r}" y1="{f.Y(0):.1f}" y2="{f.Y(0):.1f}"/>')
    body.append(f'<text class="lbl small" x="{f.w - f.r - 2}" y="{f.Y(0) - 5:.1f}" text-anchor="end">exact octave</text>')
    for k, cls, lab, ds, oi in data:
        dx = (k - (len(SPATIAL_GROUPS) - 1) / 2) * 0.13
        body.append(C.polyline(f, [(i + dx, x['mean']) for i, x in enumerate(ds)], cls))
        tail = (f'; outer minus inner third {oi["mean"]:+.3f} ± {oi["se"]:.3f}' if oi else '')
        for i, x in enumerate(ds):
            body.append(f'<line class="{cls} whisk" x1="{f.X(i + dx):.1f}" x2="{f.X(i + dx):.1f}" y1="{f.Y(x["mean"] - 2 * x["se"]):.1f}" '
                        f'y2="{f.Y(x["mean"] + 2 * x["se"]):.1f}"/>')
            body.append(C.dot(f, i + dx, x['mean'], cls, r=4, tip=f'{lab}, {RINGS[i]} third: near-wall slope {x["mean"]:+.3f} ± {x["se"]:.3f} '
                                                                  f'against the exact octave ({x["pairs"]} runs){tail}'))
    return C.svg(f, body, f'Change in near-wall slope by ring against the exact octave, {n}x{n}')


# ---------------------------------------------------------------- tables
def table_md(md_path):
    """tables.md -> HTML tables inside <details> (the generated markdown has only headings, paragraphs, lists and tables)."""
    out, rows, in_table = [], [], False

    def flush():
        nonlocal rows
        if rows:
            head, body = rows[0], rows[2:]
            out.append('<div class="scroll"><table class="data"><thead><tr>' + ''.join(f'<th>{esc(c)}</th>' for c in head) + '</tr></thead><tbody>'
                       + ''.join('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in r) + '</tr>' for r in body) + '</tbody></table></div>')
        rows = []

    for line in md_path.read_text().splitlines():
        if line.startswith('|'):
            rows.append([c.strip() for c in line.strip().strip('|').split('|')])
            in_table = True
            continue
        if in_table:
            flush()
            in_table = False
        if line.startswith('## '):
            out.append(f'<h4>{esc(line[3:])}</h4>')
        elif line.startswith('- '):
            out.append(f'<p class="li">{esc(line[2:])}</p>')
        elif line.strip() and not line.startswith('# '):
            out.append(f'<p>{esc(line)}</p>')
    flush()
    return ''.join(out)


def run_list(S):
    items = []
    for key, by_n in S['fields'].items():
        for n, v in by_n.items():
            for s, p in (v.get('paired') or {}).items():
                pass
    return items


def main():
    S = json.loads((OUT / 'summary.json').read_text())
    S['noise'] = {str(k): v for k, v in S['noise'].items()}
    S['stalls'] = {n: stall_groups(n) for n in SIZES}
    charts = dict(freq=chart_freq())
    for n in SIZES:
        charts[f'scan{n}'] = chart_scan(S, n)
        charts[f'fail{n}'] = chart_fail(S, n)
        charts[f'stall{n}'] = chart_stall(n)
        charts[f'paired{n}'] = chart_paired(S, n)
        charts[f'lock{n}'] = chart_lock(S, n)
        charts[f'conv{n}'] = chart_conv(S, n)
        charts[f'epochs{n}'] = chart_scan_epochs(S, n)
        charts[f'lossf{n}'] = chart_scan_metric(S, n, 'loss_final_mean100')
        charts[f'settle{n}'] = chart_scan_metric(S, n, 'loss_settle_epoch')
        charts[f'err{n}'] = chart_conv(S, n, 'log_err', 'median |log T/T*|, all queries', fmt=lambda v: f'{v:.2f}')
        charts[f'loss{n}'] = chart_loss(S, n)
        charts[f'cmap{n}'] = change_map(S, n)
        charts[f'spatial{n}'] = chart_spatial(S, n)
    for key, measure, xlabel, tf, vfmt in BAND_PANELS:
        charts[f'band_{key}'] = chart_band(S, '13', measure, xlabel, tf, vfmt)
    conv_legend = C.legend([(cls, lab, ('dash' if dash == '5 4' else 'dot-line') if dash else 'line') for _, cls, lab, dash in CONV_SERIES])
    epochs_legend = C.legend([(cls, lab, 'line') for _, cls, lab in EPOCH_SERIES])
    scan_legend = C.legend([('k-exact', 'exact octave (×1)', 'dot'), ('k-low', 'every row lower', 'dot'), ('k-high', 'every row higher', 'dot'),
                            ('ref', 'earlier fields, same maze', 'rule')])
    kinds = [('k-low', 'lower', 'dot'), ('k-high', 'higher', 'dot'), ('k-exact', 'nulls (same frequencies)', 'dot'), ('k-jit', 'jitter / fill', 'dot'),
             ('k-part', 'part of the band ×0.9', 'dot'), ('k-ref', 'grid-sync lock controls', 'dot')]
    paired_legend = C.legend(kinds + [('band', 'one seed: ±2 × training noise', 'band')])
    lock_legend = C.legend(kinds + [('band', '±2 × typical standard error', 'band')])
    spatial_legend = C.legend([('k-low', 'shift ×0.9 / ×0.95', 'line'), ('k-high', 'shift ×1.05 / ×1.1', 'line'),
                               ('k-jit', 'jitter ±5–20 %', 'line'), ('zero', 'exact octave (zero change)', 'rule')])
    tables = table_md(OUT / 'tables.md') if (OUT / 'tables.md').exists() else ''
    fail_legend = C.legend([('k-ref', 'reach the goal only through a wall (filled dots)', 'line'), ('k-ref', 'never reach the goal (open dots)', 'dash'),
                            ('k-low', 'every row lower', 'dot'), ('k-high', 'every row higher', 'dot')])
    stall_legend = C.legend([('k-exact', 'stopped against a wall', 'dot'), ('k-exact hollow', 'stopped in the open corridor', 'dot'),
                             ('k-low', 'every row lower', 'dot'), ('k-high', 'every row higher', 'dot')])
    metric_legend = C.legend([('k-exact', 'exact octave (×1), the reference', 'dot'), ('k-low', 'every row lower', 'dot'),
                              ('k-high', 'every row higher', 'dot')])
    page = TX.render(S=S, charts=charts, legends=dict(conv=conv_legend, epochs=epochs_legend, scan=scan_legend, paired=paired_legend, spatial=spatial_legend,
                                                      lock=lock_legend, fail=fail_legend, stall=stall_legend, metric=metric_legend),
                     tables=tables, field=lambda k, n: field(S, k, n))
    (OUT / 'octave_detune.html').write_text(page)
    print('wrote', OUT / 'octave_detune.html', len(page) // 1024, 'KB')


if __name__ == '__main__':
    main()
