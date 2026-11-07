"""Findings page for the grid-sync study (report/grid_sync.html), from the registered runs (report.collect()),
report/mechanism.json and the figures in report/.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/grid_sync/page.py

Every number on the page is computed here; the prose lives in page_template.html and page_text/*.html (hooks
<!--NAME--> are replaced by the fragments and tables built below). Publish report/grid_sync.html with the files
freq.png, speed.png, profile.png next to it.
"""
import html
import json
from pathlib import Path
import importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
_spec = importlib.util.spec_from_file_location('grid_sync_report', HERE / 'report.py')
GR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GR)
SEEDS = [87358, 1, 2]
STRICT = False          # page.py --strict (final build): every number token must resolve
LOCK = {}               # (n, control) -> lock_stats over every training seed run
BINS = GR.BINS
SIZES = GR.SIZES
FIELDS = ['ntrl-demo', 'best B × s', 'ring', 'circle', 'octave', 'octaveperm', 'octavecircle', 'literal']
LABEL = {'ntrl-demo': 'ntrl-demo', 'best B × s': 'best B × s (earlier sweep)', 'ring': 'grid-synced ring', 'circle': 'circle control',
         'literal': 'literal (cell / cos θ′)', 'scaled': 'B × s_eff', 'snap': 'B × s_eff snapped to grid', 'ringhalf': 'half ntrl-demo + half ring',
         'circlehalf': 'half ntrl-demo + half circle', 'octave': 'grid-synced octaves', 'octavecircle': 'octaves, constant magnitude (control)',
         'octaveperm': 'octaves, shuffled magnitudes (control)'}


def best_bs(by):
    """The earlier B x s run set with the highest pooled success among those with all three training seeds."""
    cand = [(f, v) for f, v in by.items() if f.startswith('B ×') and len(v) == 3]
    if not cand:
        return None, {}
    f, v = max(cand, key=lambda c: GR.pooled(c[1])[0] / GR.pooled(c[1])[1])
    return f, v


def lock_stats(t, n, ctrl):
    """Paired (same training seed) octave minus control: mean difference in points, wins, pairs; every seed run."""
    o, c = t.get(n, {}).get('octave', {}), t.get(n, {}).get(ctrl, {})
    pairs = [(o[s]['success'][0] / o[s]['success'][1] - c[s]['success'][0] / c[s]['success'][1], o[s]['success'][0] - c[s]['success'][0]) for s in o if s in c]
    if not pairs:
        return None
    return dict(points=100 * float(np.mean([p[0] for p in pairs])), wins=sum(p[1] > 0 for p in pairs), ties=sum(p[1] == 0 for p in pairs), pairs=len(pairs),
                diffs=[p[1] for p in pairs])


def table_data():
    t = GR.collect()
    t = {n: {f: {s: r for s, r in by_s.items() if s in GR.MAIN_SEEDS} for f, by_s in by.items()} for n, by in t.items()}
    out = {}
    for n in SIZES:
        by = dict(t.get(n, {}))
        name, v = best_bs(by)
        if name:
            by['best B × s'] = v
        m = {}
        for f in FIELDS:
            rs = by.get(f)
            if not rs:
                continue
            bins = [b for b in BINS if any(b in r['by_bin'] for r in rs.values())]
            rows, logerr = [], []
            for b in bins:
                per = [100 * r['by_bin'][b][0] / r['by_bin'][b][1] for r in rs.values() if b in r['by_bin']]
                a = sum(r['by_bin'][b][0] for r in rs.values() if b in r['by_bin'])
                c = sum(r['by_bin'][b][1] for r in rs.values() if b in r['by_bin'])
                rows.append([round(float(np.mean(per)), 1), round(min(per), 1), round(max(per), 1), len(per), a, c])
                logerr.append(round(float(np.mean([r['log_err_by_bin'][b] for r in rs.values() if b in r['log_err_by_bin']])), 3))
            k, mm, per = GR.pooled(rs)
            m[f] = dict(bins=bins, rows=rows, logerr=logerr, total=[k, mm], per_seed={str(s): rs[s]['success'] for s in rs},
                        speed=round(float(np.mean([r['speed_corr'] for r in rs.values()])), 3),
                        err=round(float(np.mean([r['log_err'] for r in rs.values()])), 3),
                        cos=round(float(np.mean([r['descent_cos'] for r in rs.values()])), 3),
                        stalls=int(sum(r['success'][1] - r['converged'] for r in rs.values())),
                        hits=int(sum(r['converged'] - r['success'][0] for r in rs.values())),
                        ids={str(s): rs[s]['id'] for s in rs}, label=name if f == 'best B × s' else LABEL[f])
        out[str(n)] = m
    return out


def pct(a, b, d=0):
    return f'{100 * a / b:.{d}f}' if b else '–'


def totals_table(data):
    head = '<table class="named"><thead><tr><th>field</th><th>features</th>' + ''.join(f'<th>{n}×{n}</th>' for n in SIZES) + '</tr></thead><tbody>'
    rows = []
    for f in FIELDS:
        cells = []
        for n in SIZES:
            d = data[str(n)].get(f)
            if d is None:
                cells.append('<td class="num">–</td>')
                continue
            per = [d['per_seed'][str(s)][0] for s in SEEDS if str(s) in d['per_seed']]
            sub = f'<br><span class="mute">{" / ".join(map(str, per))} of {d["per_seed"][next(iter(d["per_seed"]))][1]}</span>' if len(per) > 1 else \
                f'<br><span class="mute">{per[0]} of {d["per_seed"][next(iter(d["per_seed"]))][1]}, seed 87358</span>'
            lab = f' <span class="mute">({html.escape(d["label"])})</span>' if f == 'best B × s' else ''
            cells.append(f'<td class="num"><b>{pct(*d["total"])} %</b>{lab}{sub}</td>')
        rows.append(f'<tr><td>{html.escape(LABEL[f])}</td><td>{html.escape(DESC.get(f, ""))}</td>' + ''.join(cells) + '</tr>')
    return head + ''.join(rows) + '</tbody></table>'


DESC = {'ntrl-demo': 'B ~ N(0, 1), as shipped', 'best B × s': 'B × 1.5 / 3 / 6, best per maze', 'ring': '|f| = f_grid / cos θ′ (in sync)',
        'circle': '|f| = 1.14 f_grid (same mean, not in sync)', 'literal': '|f| = f_grid · cos θ′', 'scaled': 'B × s_eff, s_eff ≈ E / 1.9',
        'snap': 'B × s_eff, rows ≥ f_grid/2 on grid harmonics', 'ringhalf': 'even rows ntrl-demo, odd rows ring', 'circlehalf': 'even rows ntrl-demo, odd rows circle',
        'octave': 'half periods 1, 2, 4, … cells, each f_grid / (L cos θ′)', 'octavecircle': 'the same octaves, constant |f| per octave',
        'octaveperm': 'the same magnitudes and directions, shuffled within each octave'}


def accuracy_table(data):
    head = '<table class="named"><thead><tr><th>field</th><th>seeds</th>' + ''.join(f'<th>{n}×{n}</th>' for n in SIZES) + '</tr></thead><tbody>'
    rows = []
    for f in FIELDS:
        cells, seeds = [], set()
        for n in SIZES:
            d = data[str(n)].get(f)
            if d is None:
                cells.append('<td class="num">–</td>')
                continue
            seeds.add(len(d['per_seed']))
            cells.append(f'<td class="num">{d["err"]:.3f}<br><span class="mute">fit {d["speed"]:.2f} · cos {d["cos"]:.2f}</span></td>')
        rows.append(f'<tr><td>{html.escape(LABEL[f])}</td><td class="num">{"/".join(map(str, sorted(seeds)))}</td>' + ''.join(cells) + '</tr>')
    return head + ''.join(rows) + '</tbody></table>'


def mechanism_table():
    p = OUT / 'mechanism.json'
    if not p.exists():
        return ''
    mech = json.loads(p.read_text())
    order = ['ntrl-demo', 'B × 1.5', 'B × 3', 'B × 6'] + GR.MODES
    head = ('<table class="named"><thead><tr><th>maze</th><th>field</th><th>doorway dip</th><th>phantom doorways</th><th>speed 0.35 cell from a wall</th>'
            '<th>bumps per long path</th><th>near-wall slope / cos</th></tr></thead><tbody>')
    rows = []
    for r in sorted(mech.values(), key=lambda r: (r['n'], order.index(r['field']) if r['field'] in order else 99)):
        if r['seed'] != 87358:
            continue
        pr, w = r['profiles'], r['wall'].get('2-8')
        rows.append(f'<tr><td>{r["n"]}×{r["n"]}</td><td>{html.escape(LABEL.get(r["field"], r["field"]))}</td><td class="num">{pr["doorway_dip_median"]:.2f}</td>'
                    f'<td class="num">{100 * pr["doorway_phantom_share"]:.0f} %</td><td class="num">{pr["wall_S_at_035"]:.2f} <span class="mute">(truth {pr["wall_label_at_035"]:.2f})</span></td>'
                    f'<td class="num">{r["bumps"]["bumps_per_path"]:.2f}</td><td class="num">' + (f'{w["slope"]:.2f} / {w["cos"]:.2f}' if w else '–') + '</td></tr>')
    return head + ''.join(rows) + '</tbody></table>'


def lock_table(full):
    head = ('<table><thead><tr><th>maze</th><th>training seed</th><th>synced octaves</th><th>shuffled magnitudes</th><th>constant magnitude</th>'
            '<th>synced − shuffled</th><th>synced − constant</th></tr></thead><tbody>')
    rows = []
    for n in SIZES:
        o, pm, c = (full.get(n, {}).get(f, {}) for f in ('octave', 'octaveperm', 'octavecircle'))
        for s in sorted(o, key=lambda s: (s != 87358, s)):
            a, m = o[s]['success']
            b = pm.get(s, {}).get('success', [None])[0]
            d = c.get(s, {}).get('success', [None])[0]
            f = lambda v: '–' if v is None else str(v)
            df = lambda v: '–' if v is None else f'{a - v:+d}'.replace('-', '−')
            rows.append(f'<tr><td>{n}×{n}</td><td class="num">{s}</td><td class="num">{a} <span class="mute">of {m}</span></td><td class="num">{f(b)}</td>'
                        f'<td class="num">{f(d)}</td><td class="num">{df(b)}</td><td class="num">{df(d)}</td></tr>')
    return head + ''.join(rows) + '</tbody></table>'


def tiles(data):
    out = []
    for n in SIZES:
        m = data[str(n)]
        o, b, bs = m.get('octave'), m.get('ntrl-demo'), m.get('best B × s')
        if o is None or b is None:
            continue
        seeds = len(o['per_seed'])
        best = f' · best B × s {pct(*bs["total"])} %' if bs else ''
        out.append(f'<div class="tile"><div class="k">{n} × {n} maze</div><div class="v">{pct(*o["total"])} % <small>octaves</small></div>'
                   f'<div class="s">ntrl-demo {pct(*b["total"])} %{best}</div><div class="s mute">grid-synced octaves, {seeds} training seed{"s" if seeds > 1 else ""}</div></div>')
    return ''.join(out)


def runs_table(data):
    rows = []
    for n in SIZES:
        for f in FIELDS:
            d = data[str(n)].get(f)
            if d is None:
                continue
            for s, i in d['ids'].items():
                rows.append(f'<tr><td>{n}×{n}</td><td>{html.escape(d["label"])}</td><td class="num">{s}</td><td class="num">{i}</td></tr>')
    return '<table class="named"><thead><tr><th>maze</th><th>field</th><th>training seed</th><th>run id</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'


def fill(text, data, mech):
    """Replace {{kind:field:n}} tokens with numbers from the runs: succ (pooled %), runs (k/m), per (per-seed counts),
    seeds, fit (speed fit), err (|log T/T*|); slope, cos, bumps, wallS, dip (mechanism.json, seed 87358)."""
    import re

    def one(m):
        kind, f, n = m.group(1), m.group(2), m.group(3)
        if kind in ('lock', 'wins'):
            st = LOCK.get((int(n), f))
            if st is None:
                if STRICT:
                    raise KeyError(f'no lock pairs for {f} on {n}x{n}')
                return '[pending]'
            return f'{st["points"]:.1f}'.replace('-', '−') if kind == 'lock' else f'{st["wins"]} of {st["pairs"]}'
        if kind in ('slope', 'cos', 'bumps', 'wallS', 'dip'):
            r = next((v for v in mech.values() if v['n'] == int(n) and v['field'] == f and v['seed'] == 87358), None)
            if r is None:
                raise KeyError(f'no mechanism entry for {f} on {n}x{n} (run mechanism.py)')
            w = r['wall'].get('2-8', {})
            return {'slope': f'{w.get("slope", float("nan")):.2f}', 'cos': f'{w.get("cos", float("nan")):.2f}', 'bumps': f'{r["bumps"]["bumps_per_path"]:.2f}',
                    'wallS': f'{r["profiles"]["wall_S_at_035"]:.2f}', 'dip': f'{r["profiles"]["doorway_dip_median"]:.2f}'}[kind]
        d = data.get(n, {}).get(f)
        if d is None:
            if STRICT:
                raise KeyError(f'no runs for {f} on {n}x{n}')
            return '[pending]'
        k, mm = d['total']
        if kind == 'succ':
            return f'{100 * k / mm:.0f} %'
        if kind == 'runs':
            return f'{k:,}/{mm:,}'
        if kind == 'per':
            return ' / '.join(str(d['per_seed'][str(s)][0]) for s in SEEDS if str(s) in d['per_seed'])
        if kind == 'seeds':
            return str(len(d['per_seed']))
        if kind == 'fit':
            return f'{d["speed"]:.2f}'
        if kind == 'err':
            return f'{d["err"]:.2f}'
        return '?'
    out = re.sub(r'\{\{(\w+):([^:}]+):(\d+)\}\}', one, text)
    assert '{{' not in out, 'unfilled token'
    return out


def main():
    global STRICT
    import sys
    STRICT = '--strict' in sys.argv
    full = GR.collect()
    for n in SIZES:
        for ctrl in ('octaveperm', 'octavecircle'):
            st = lock_stats(full, n, ctrl)
            if st:
                LOCK[(n, ctrl)] = st
    data = table_data()
    mech = json.loads((OUT / 'mechanism.json').read_text()) if (OUT / 'mechanism.json').exists() else {}
    tpl = (HERE / 'page_template.html').read_text()
    for frag in sorted((HERE / 'page_text').glob('*.html')):
        tpl = tpl.replace(f'<!--{frag.stem.upper().replace(".SVG", "")}-->', fill(frag.read_text(), data, mech))
    tpl = tpl.replace('<!--TILES-->', tiles(data)).replace('<!--TOTALS-->', totals_table(data)).replace('<!--ACCURACY-->', accuracy_table(data))
    tpl = tpl.replace('<!--MECHANISM-->', mechanism_table()).replace('<!--RUNS-->', runs_table(data)).replace('<!--LOCKTABLE-->', lock_table(full))
    tpl = tpl.replace('/*DATA*/', json.dumps(dict(series=data, sizes=[str(n) for n in SIZES], fields=FIELDS, label=LABEL), separators=(',', ':')))
    (OUT / 'grid_sync.html').write_text(tpl)
    (OUT / 'data.json').write_text(json.dumps(data, indent=1))
    print('wrote', OUT / 'grid_sync.html', len(tpl))


if __name__ == '__main__':
    main()
