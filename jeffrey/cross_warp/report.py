"""Tables for the cross-warp study from the registered runs: report/tables.md and report/summary.json.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/report.py [--labels a,b,...]

Rows: ntrl-demo on the plus maze (maze_scale plus-ntrl stage, the baseline) and every cross-warp variant (label = the
registry config's label), pooled over the training seeds each has; the 'paired' columns compare a variant with
ntrl-demo run by run (same query, planning seed and training seed): runs only the variant solves / only ntrl-demo
solves, and the exact two-sided McNemar p. Success = MPPI successes over planning seeds 0/1/2.
"""
import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'report'
REGISTRY_RUNS = Path(os.environ.get('NTFIELDS_REGISTRY', Path.home() / '.local/share/ntfields-experiments')) / 'runs'
BINS = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64']
SUPERSEDED = set()


def label_of(rid):
    spec = REGISTRY_RUNS / rid / 'spec.json'
    if spec.exists():
        return json.loads(spec.read_text())['config'].get('label')
    return None


def collect():
    base, warp = {}, defaultdict(dict)
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        rid = r.parent.name.replace('registered_', '')
        if rid in SUPERSEDED:
            continue
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        if not d.get('complete') or d.get('smoke'):
            continue
        d['_id'], d['_dir'] = rid, r.parent
        if d.get('stage') == 'plus-ntrl':
            base[(d['prepared'], d['seed'])] = d
        elif d.get('stage') == 'cross-warp':
            lab = label_of(rid)
            if lab is None:
                continue
            key = (lab, d['prepared'])
            if d['seed'] in warp[key]:            # a repeat: keep the later run
                pass
            warp[key][d['seed']] = d
    return base, warp


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def pooled(by_seed):
    seeds = sorted(by_seed, key=lambda s: (s != 87358, s))
    tot = [by_seed[s]['success_total'] for s in seeds]
    per_bin = []
    for l in BINS:
        v = [by_seed[s]['success_by_bin'].get(l) for s in seeds]
        v = [x for x in v if x]
        per_bin.append(100 * sum(x[0] for x in v) / sum(x[1] for x in v) if v else None)
    row = dict(seeds=seeds, per_seed=[t[0] for t in tot], of=tot[0][1], total=sum(t[0] for t in tot), runs=sum(t[1] for t in tot), per_bin=per_bin,
               ids=[by_seed[s]['_id'] for s in seeds])
    row['pct'] = 100 * row['total'] / row['runs']
    short = [by_seed[s]['success_by_bin'] for s in seeds]
    row['upto32'] = sum(sum(v[0] for k, v in b.items() if k != '32-64') for b in short) / len(seeds)     # per seed, of 600
    succ = np.concatenate([np.load(by_seed[s]['_dir'] / 'rows.npz')['success'] for s in seeds])
    col = np.concatenate([np.load(by_seed[s]['_dir'] / 'rows.npz')['collision'] for s in seeds])
    conv = np.concatenate([np.load(by_seed[s]['_dir'] / 'rows.npz')['converged'] for s in seeds])
    row['collision'] = int(col.sum())
    row['stalled'] = int((~conv).sum())
    d0 = by_seed[seeds[0]]
    sf = [by_seed[s].get('speed_fit', {}).get('log_speed_corr') for s in seeds]
    row['speed_fit'] = float(np.mean([x for x in sf if x is not None])) if any(x is not None for x in sf) else None
    row['field_tt'] = float(np.median([by_seed[s]['field']['t_over_tstar_median'] for s in seeds]))
    nw = [by_seed[s].get('near_wall', {}).get('0.5-2') for s in seeds]
    nw = [x for x in nw if x]
    row['near_wall'] = (float(np.mean([x['slope'] for x in nw])), float(np.mean([x['cos'] for x in nw]))) if nw else None
    bp = [by_seed[s].get('bumps', {}).get('bumps_per_path') for s in seeds]
    bp = [x for x in bp if x is not None]
    row['bumps'] = float(np.mean(bp)) if bp else None
    pr = [by_seed[s].get('profiles') for s in seeds]
    pr = [x for x in pr if x]
    row['doorway_dip'] = float(np.mean([x['doorway_dip_median'] for x in pr])) if pr else None
    row['wall_S035'] = float(np.mean([x['wall_S_at_035'] for x in pr])) if pr else None
    bw = [by_seed[s].get('by_where') for s in seeds]
    if all(bw):
        row['by_where'] = {k: 100 * sum(x[k]['success'] for x in bw) / sum(x[k]['runs'] for x in bw) for k in ('both', 'one', 'neither')}
    ws = [by_seed[s].get('warp_stats') for s in seeds]
    ws = [x for x in ws if x]
    if ws and 'plus' in ws[0]:
        row['stretch'] = dict(plus=float(np.mean([math.exp(x['plus']['logdet_median'] / 2) for x in ws])),
                              quadrants=float(np.mean([math.exp(x['quadrants']['logdet_median'] / 2) for x in ws])),
                              near_wall=float(np.mean([math.exp(x['near_wall']['logdet_median'] / 2) for x in ws])),
                              heading_quadrants=float(np.mean([math.exp(x['quadrants']['heading_median']) for x in ws])),
                              heading_plus=float(np.mean([math.exp(x['plus']['heading_median']) for x in ws])),
                              sv=(min(x['sv_xy']['min'] for x in ws), max(x['sv_xy']['max'] for x in ws)))
    row['train_min'] = float(np.mean([by_seed[s].get('train_seconds', 0) / 60 for s in seeds]))
    row['_succ'] = succ
    return row


def table(base, warp, labels=None):
    lines = ['| field | seeds | runs solved per seed (of 720) | total % | ≤ 32 cells, per seed (of 600) | ' + ' | '.join(BINS) + ' | both / one / neither in + | collisions | stalls | paired vs ntrl-demo (+ / − / p) | speed fit | near-wall slope / cos | bumps / long path | doorway dip | stretch + / quadrants / near wall (linear) | train min |',
             '|' + '---|' * (14 + len(BINS))]
    summary = {}
    preps = sorted({p for p, _ in base} | {p for _, p in warp})
    for prep in preps:
        bseeds = {s: d for (p, s), d in base.items() if p == prep}
        rows = []
        if bseeds:
            rows.append(('ntrl-demo', bseeds))
        for (lab, p), by in sorted(warp.items()):
            if p == prep and (labels is None or lab in labels):
                rows.append((lab, by))
        for name, by in rows:
            r = pooled(by)
            paired = ''
            if name != 'ntrl-demo':
                common = [s for s in r['seeds'] if s in bseeds]
                if common:
                    a = np.concatenate([np.load(by[s]['_dir'] / 'rows.npz')['success'] for s in common])
                    b = np.concatenate([np.load(bseeds[s]['_dir'] / 'rows.npz')['success'] for s in common])
                    gain, loss = int((a & ~b).sum()), int((~a & b).sum())
                    r['paired'] = dict(seeds=common, gain=gain, loss=loss, p=mcnemar(gain, loss))
                    paired = f'+{gain} / −{loss} / {mcnemar(gain, loss):.2g}'
            fmt = lambda v, f='{:.0f}': '–' if v is None else f.format(v)
            bw = r.get('by_where')
            st = r.get('stretch')
            lines.append(f'| {name} | {len(r["seeds"])} | {" / ".join(map(str, r["per_seed"]))} | {r["pct"]:.1f} | {r["upto32"]:.0f} | ' + ' | '.join(fmt(v) for v in r['per_bin']) +
                         f' | {fmt(bw["both"]) + " / " + fmt(bw["one"]) + " / " + fmt(bw["neither"]) if bw else "–"} | {r["collision"]} | {r["stalled"]} | {paired or "–"} | '
                         f'{fmt(r["speed_fit"], "{:.2f}")} | {(fmt(r["near_wall"][0], "{:.2f}") + " / " + fmt(r["near_wall"][1], "{:.2f}")) if r["near_wall"] else "–"} | '
                         f'{fmt(r["bumps"], "{:.2f}")} | {fmt(r["doorway_dip"], "{:.2f}")} | '
                         f'{(f"{st["plus"]:.2f} / {st["quadrants"]:.2f} / {st["near_wall"]:.2f}") if st else "–"} | {r["train_min"]:.1f} |')
            r.pop('_succ')
            summary[f'{name}|{prep}'] = r
    return lines, summary


def collect_evalq():
    """{query set: {run id: entry}} from every complete evalq run (later runs override earlier ones for the same field)."""
    out = defaultdict(dict)
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        if d.get('stage') != 'evalq' or d.get('smoke') or not d.get('complete'):
            continue
        q = Path(d['queries']).name
        for rid, e in d['runs'].items():
            e['_eval'] = r.parent
            out[q][rid] = e
    return out


def evalq_table(base, warp, labels=None):
    """Success on each extra query set, pooled over the seeds a field has there (planning seed(s) of that evaluation)."""
    lines, summary = [], {}
    ev = collect_evalq()
    names = {}
    for (p, s), d in base.items():
        names[d['_id']] = ('ntrl-demo', s)
    for (lab, p), by in warp.items():
        for s, d in by.items():
            names[d['_id']] = (lab, s)
    for q, entries in sorted(ev.items()):
        groups = defaultdict(dict)
        for rid, e in entries.items():
            if rid in names:
                lab, seed = names[rid]
                groups[lab][seed] = e
        lines += [f'### Query set {q}', '', '| field | seeds | solved per seed | total % | ≤ 32 cells % | ' + ' | '.join(BINS) + ' | both / one / neither in + |', '|' + '---|' * (6 + len(BINS))]
        order = ['ntrl-demo', 'id'] + sorted(k for k in groups if k not in ('ntrl-demo', 'id'))
        for lab in order:
            if lab not in groups or (labels is not None and lab not in labels and lab != 'ntrl-demo'):
                continue
            by = groups[lab]
            seeds = sorted(by, key=lambda x: (x != 87358, x))
            tot = [by[x]['success_total'] for x in seeds]
            pb = []
            for l in BINS:
                v = [by[x]['success_by_bin'].get(l) for x in seeds]
                v = [x for x in v if x]
                pb.append(100 * sum(x[0] for x in v) / sum(x[1] for x in v) if v else None)
            short = [sum(v[0] for k, v in by[x]['success_by_bin'].items() if k != '32-64') for x in seeds]
            short_n = [sum(v[1] for k, v in by[x]['success_by_bin'].items() if k != '32-64') for x in seeds]
            bw = [by[x].get('by_where') for x in seeds]
            where = ' / '.join(f'{100 * sum(b[k][0] for b in bw) / sum(b[k][1] for b in bw):.0f}' for k in ('both', 'one', 'neither')) if all(bw) else '–'
            row = dict(seeds=seeds, per_seed=[t[0] for t in tot], of=tot[0][1], pct=100 * sum(t[0] for t in tot) / sum(t[1] for t in tot),
                       short_pct=100 * sum(short) / sum(short_n), per_bin=pb)
            summary[f'{lab}|{q}'] = row
            lines.append(f'| {lab} | {len(seeds)} | {" / ".join(map(str, row["per_seed"]))} (of {row["of"]}) | {row["pct"]:.1f} | {row["short_pct"]:.1f} | '
                         + ' | '.join('–' if v is None else f'{v:.0f}' for v in pb) + f' | {where} |')
        lines.append('')
    return lines, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', default='')
    a = p.parse_args()
    base, warp = collect()
    labels = set(a.labels.split(',')) if a.labels else None
    lines, summary = table(base, warp, labels)
    elines, esummary = evalq_table(base, warp, labels)
    summary['_evalq'] = esummary
    OUT.mkdir(exist_ok=True)
    (OUT / 'tables.md').write_text('# Cross warp: tables (generated by report.py)\n\n## Prepared queries (240, planning seeds 0/1/2)\n\n' + '\n'.join(lines)
                                   + '\n\n## Second query set\n\n' + '\n'.join(elines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print('\n'.join(lines))
    print('\n'.join(elines))


if __name__ == '__main__':
    main()
