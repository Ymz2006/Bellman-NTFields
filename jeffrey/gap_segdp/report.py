"""Tables for segment DP on the gap mazes, from registered runs: the DP chains (stage gap-segdp3 / gap-segdp) next to
the gap study's fields on the same queries (jeffrey/cross_warp gapreport.collect(): ntrl-demo = the DP chain's base,
and the warps).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_segdp/report.py

Writes report/tables.md and report/summary.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))
import gapreport as GR  # noqa: E402

KS = [1, 2, 4, 8, 12]
SEEDS = [87358, 1, 2]
CATS = GR.CATS
SHORT = {'same quadrant': 'same', 'adjacent quadrants': 'adjacent', 'diagonal quadrants': 'diagonal', 'quadrant - open': 'quad–open', 'open - open': 'open–open'}
WARPS = ['constructed', 'warp + scaling', 'README warp']
OUT = HERE / 'report'


def dp_runs():
    out = defaultdict(lambda: defaultdict(dict))
    for f in sorted((ROOT / 'results').glob('registered_2026110[4-9]*/results.json')):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not str(d.get('stage', '')).startswith('gap-segdp') or d.get('smoke') or not d.get('complete'):
            continue
        w = d['config'].get('normal_weight') if d.get('method') == 'segdp3' else None
        name = 'segment DP' + (f' + wall term {w:g}' if w else ' (plain)')
        d['_id'] = f.parent.name.replace('registered_', '')
        out[d['gap_scale']][name][d['seed']] = d
    return out


def pct(a, b):
    return 100 * a / b if b else float('nan')


def main():
    gap, _ = GR.collect()
    dp = dp_runs()
    md = ['# Segment DP on the gap mazes', '',
          'MPPI success (%), 300 queries × planning seeds 0/1/2 per training seed. ntrl-demo is the base every DP chain starts from; '
          'warps from the gap study for reference.', '']
    summary = {}
    # 1. totals per seed
    md += ['## All queries, per training seed', '', '| gap | field | seed 87358 | seed 1 | seed 2 | pooled |', '|---|---|---|---|---|---|']
    for k in KS:
        rows = [('ntrl-demo', gap[k].get('ntrl-demo', {}))] + [(n, {s: dict(success_total=d['success_total']) for s, d in by.items()}) for n, by in sorted(dp[k].items())] \
            + [(w, gap[k].get(w, {})) for w in WARPS if gap[k].get(w)]
        for name, by in rows:
            if not by:
                continue
            cells = [f'{pct(*by[s]["success_total"]):.1f}' if s in by else '' for s in SEEDS]
            A = sum(by[s]['success_total'][0] for s in by)
            B = sum(by[s]['success_total'][1] for s in by)
            md.append(f'| ×{k} | {name} | ' + ' | '.join(cells) + f' | {pct(A, B):.1f} |')
            summary.setdefault(str(k), {})[name] = dict(per_seed={str(s): by[s]['success_total'] for s in by}, pooled=[A, B])
    md.append('')
    # 2. by category, pooled over the seeds both have
    md += ['## By region pair (pooled over training seeds)', '', '| gap | field | seeds | ' + ' | '.join(SHORT[c] for c in CATS) + ' |', '|---|---|---|' + '---|' * len(CATS)]
    for k in KS:
        rows = [('ntrl-demo', {s: e['by_category'] for s, e in gap[k].get('ntrl-demo', {}).items()})]
        rows += [(n, {s: {c: [v['success'], v['runs']] for c, v in d['by_category'].items()} for s, d in by.items()}) for n, by in sorted(dp[k].items())]
        rows += [(w, {s: e['by_category'] for s, e in gap[k].get(w, {}).items()}) for w in WARPS if gap[k].get(w)]
        for name, by in rows:
            if not by:
                continue
            cells = []
            for c in CATS:
                v = [by[s][c] for s in by if c in by[s]]
                cells.append(f'{pct(sum(x[0] for x in v), sum(x[1] for x in v)):.0f}' if v else '')
            md.append(f'| ×{k} | {name} | {len(by)} | ' + ' | '.join(cells) + ' |')
            summary[str(k)].setdefault(name, {})['by_category'] = {c: [sum(by[s][c][0] for s in by if c in by[s]), sum(by[s][c][1] for s in by if c in by[s])] for c in CATS}
    md.append('')
    # 3. DP failure modes and field diagnostics, per run
    md += ['## Segment DP runs: failures and field', '',
           'Failures out of 900 runs (a run can both stall and collide). T/T*: median learned over true travel time on the queries; '
           'top-level target/T*: the last DP level\'s targets over the truth on probe pairs; base/T* the same probes on the base.', '',
           '| gap | seed | run | success | collisions | stalls without collision | T/T* | Spearman | descent cos | levels | top target/T* | base/T* there | train min |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in KS:
        for name, by in sorted(dp[k].items()):
            for s in SEEDS:
                d = by.get(s)
                if not d:
                    continue
                f, fl = d['field'], d['failures']
                top = [x for x in d['level_stats'] if x.get('probe')]
                tp = top[-1]['probe'] if top else {}
                md.append(f'| ×{k} | {s} | `{d["_id"]}` | {d["success_total"][0]} | {fl["collision"]} | {fl["stalled_no_collision"]} | {f["t_over_tstar_median"]:.2f} | '
                          f'{f["spearman"]:.3f} | {f["descent_cos_median"]:.2f} | {d["levels"]} | {tp.get("target_over_truth", {}).get("median", float("nan")):.2f} | '
                          f'{tp.get("base_over_truth", {}).get("median", float("nan")):.2f} | {d["elapsed_seconds"] / 60:.0f} |')
    md.append('')
    OUT.mkdir(exist_ok=True)
    (OUT / 'tables.md').write_text('\n'.join(md) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
