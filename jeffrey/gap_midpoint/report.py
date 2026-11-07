"""Tables for the midpoint loss on the gap mazes, from registered runs (stage gap-midpoint), next to ntrl-demo (the base),
the same resumed loop without the term (control), segment DP (jeffrey/gap_segdp) and the gap study's warps.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_midpoint/report.py

Writes report/tables.md and report/summary.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))
sys.path.insert(0, str(ROOT / 'jeffrey/gap_segdp'))
import gapreport as GR  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location('segreport', ROOT / 'jeffrey/gap_segdp/report.py')
SR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SR)

KS = [1, 2, 4, 8, 12]
SEEDS = [87358, 1, 2]
CATS = GR.CATS
SHORT = SR.SHORT
WARPS = ['constructed', 'warp + scaling', 'README warp']
OUT = HERE / 'report'


def mid_runs():
    out = defaultdict(lambda: defaultdict(dict))
    for f in sorted((ROOT / 'results').glob('registered_2026110[4-9]*/results.json')):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get('stage') != 'gap-midpoint' or d.get('smoke') or not d.get('complete'):
            continue
        c = d['config']
        name = 'control (resumed, no term)' if c['weight'] == 0 else 'midpoint loss' if (c['leaf'], c['traverse'], c['nodes'], c.get('ends', 'free')) == ('min', 'safe', 'both', 'free') and c['weight'] == 1 else None
        if name is None:
            continue
        d['_id'] = f.parent.name.replace('registered_', '')
        out[d['gap_scale']][name][d['seed']] = d
    return out


def pct(a, b):
    return 100 * a / b if b else float('nan')


def main():
    gap, _ = GR.collect()
    mid, seg = mid_runs(), SR.dp_runs()
    md = ['# Midpoint loss on the gap mazes', '',
          'MPPI success (%), 300 queries × planning seeds 0/1/2 per training seed. Same queries for every row.', '']
    summary = {}

    def rows_total(k):
        rows = [('ntrl-demo', gap[k].get('ntrl-demo', {}))]
        rows += [(n, {s: dict(success_total=d['success_total']) for s, d in mid[k].get(n, {}).items()}) for n in ('control (resumed, no term)', 'midpoint loss')]
        rows += [(n, {s: dict(success_total=d['success_total']) for s, d in by.items()}) for n, by in sorted(seg[k].items())]
        return rows + [(w, gap[k].get(w, {})) for w in WARPS if gap[k].get(w)]

    def rows_cat(k):
        conv = lambda by: {s: {c: [v['success'], v['runs']] for c, v in d['by_category'].items()} for s, d in by.items()}
        rows = [('ntrl-demo', {s: e['by_category'] for s, e in gap[k].get('ntrl-demo', {}).items()})]
        rows += [(n, conv(mid[k].get(n, {}))) for n in ('control (resumed, no term)', 'midpoint loss')]
        rows += [(n, conv(by)) for n, by in sorted(seg[k].items())]
        return rows + [(w, {s: e['by_category'] for s, e in gap[k].get(w, {}).items()}) for w in WARPS if gap[k].get(w)]

    md += ['## All queries, per training seed', '', '| gap | field | seed 87358 | seed 1 | seed 2 | pooled |', '|---|---|---|---|---|---|']
    for k in KS:
        for name, by in rows_total(k):
            if not by:
                continue
            cells = [f'{pct(*by[s]["success_total"]):.1f}' if s in by else '' for s in SEEDS]
            A, B = sum(by[s]['success_total'][0] for s in by), sum(by[s]['success_total'][1] for s in by)
            md.append(f'| ×{k} | {name} | ' + ' | '.join(cells) + f' | {pct(A, B):.1f} |')
            summary.setdefault(str(k), {})[name] = dict(per_seed={str(s): by[s]['success_total'] for s in by}, pooled=[A, B])
    md += ['', '## By region pair (pooled over the training seeds listed)', '', '| gap | field | seeds | ' + ' | '.join(SHORT[c] for c in CATS) + ' |', '|---|---|---|' + '---|' * len(CATS)]
    for k in KS:
        for name, by in rows_cat(k):
            if not by:
                continue
            cells = []
            for c in CATS:
                v = [by[s][c] for s in by if c in by[s]]
                cells.append(f'{pct(sum(x[0] for x in v), sum(x[1] for x in v)):.0f}' if v else '')
            md.append(f'| ×{k} | {name} | {len(by)} | ' + ' | '.join(cells) + ' |')
            summary[str(k)].setdefault(name, {})['by_category'] = {c: [sum(by[s][c][0] for s in by if c in by[s]), sum(by[s][c][1] for s in by if c in by[s])] for c in CATS}
    md += ['', '## Runs: failures, field and training drives', '',
           'Failures out of 900 runs. T/T*: median learned over true travel time on the queries. Last refresh: drives reaching their goal of 8,192, and mean steps.', '',
           '| gap | field | seed | run | success | collisions | stalls without collision | T/T* | Spearman | descent cos | last refresh reached | steps | min |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in KS:
        for name in ('control (resumed, no term)', 'midpoint loss'):
            for s in SEEDS:
                d = mid[k].get(name, {}).get(s)
                if not d:
                    continue
                f, fl = d['field'], d['failures']
                r = d['refresh'][-1] if d.get('refresh') else {}
                md.append(f'| ×{k} | {name} | {s} | `{d["_id"]}` | {d["success_total"][0]} | {fl["collision"]} | {fl["stalled_no_collision"]} | {f["t_over_tstar_median"]:.2f} | '
                          f'{f["spearman"]:.3f} | {f["descent_cos_median"]:.2f} | {r.get("reached", "")} | {r.get("steps_mean", 0):.0f} | {d["elapsed_seconds"] / 60:.0f} |')
    md.append('')
    OUT.mkdir(exist_ok=True)
    (OUT / 'tables.md').write_text('\n'.join(md) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
